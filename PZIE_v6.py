import numpy as np
import scipy.optimize as opt
from typing import Optional, Tuple

class ZeemanLinearParabolicInversion:
    """
    Joint inversion of ionospheric current systems using linear parabolic fitting.
    
    This approach models each spectrum as an inverted parabola with:
    - Center fixed by the physics model: B_peak = sqrt(2*B0*dB_parallel + B0²)
    - Width and amplitude determined by linear least squares fitting
    
    The key insight: Using shifted coordinates, the inverted parabola becomes
    a standard polynomial that's linear in its coefficients, enabling fast fitting.
    
    Forward model:
    1. Current system m → magnetic field perturbation: dB_parallel = G*m
    2. Total field: B_total = sqrt(2*B0*dB_parallel + B0²)
    3. For each spectrum: fit inverted parabola with center at B_total
    4. Misfit is residual between observed and fitted spectra
    """
    
    def __init__(self, 
                 G: np.ndarray, 
                 B0_grid: np.ndarray,
                 altitude_grid: np.ndarray,
                 time_series: np.ndarray, 
                 xB: np.ndarray,
                 mlat: np.ndarray,
                 l1: float = 1e-6,
                 l2: float = 0.0,
                 LL: Optional[np.ndarray] = None,
                 m0: Optional[np.ndarray] = None,
                 dhp0: float = 0.0,
                 ah0: float = 0.0):
        """
        Initialize the linear parabolic inversion.
        
        Parameters:
        -----------
        G : np.ndarray
            Green's function matrix (n_obs x n_model) - computes dB_parallel
        B0_grid : np.ndarray
            Pre-computed B0 values at different altitudes (n_obs x n_altitudes)
        altitude_grid : np.ndarray
            Altitude grid corresponding to B0_grid columns (n_altitudes,) in m
        time_series : np.ndarray
            Zeeman split time series data (n_obs x n_spectra x n_spectral_points)
        xB : np.ndarray
            Magnetic field axis for each spectrum (n_obs x n_spectral_points)
        mlat : np.ndarray
            Magnetic latitude for each observation (n_obs,) in degrees
        l1 : float
            0th order (damping) regularization parameter
        l2 : float
            1st order (smoothing) regularization parameter
        LL : np.ndarray, optional
            ??
        m0 : np.ndarray, optional
            Initial guess for model parameters
        dhp0 : float
            Initial guess for altitude correction at pole in m (default 0.0)
        ah0 : float
            Initial guess for altitude gradient in m/degree (default 0.0)
        """
        self.G = G
        self.B0_grid = B0_grid
        self.altitude_grid = altitude_grid
        self.time_series = time_series
        self.xB = xB
        self.mlat = mlat
        self.l1 = l1
        self.l2 = l2
        self.dhp = dhp0  # Altitude correction at pole
        self.ah = ah0    # Altitude gradient
        
        self.n_obs, self.n_model = G.shape
        _, self.n_spectra, self.n_spectral_points = time_series.shape
        self.n_altitude = self.altitude_grid.size
        
        # Reference altitude (assumed)
        self.h_ref = 80.0e3  # m
        
        # Pre-compute B0 derivatives w.r.t. altitude
        self.dB0_dh = np.zeros((self.n_obs, self.n_altitude - 1))
        for i in range(self.n_altitude - 1):
            dh_grid = self.altitude_grid[i+1] - self.altitude_grid[i]
            self.dB0_dh[:, i] = (self.B0_grid[:, i+1] - self.B0_grid[:, i]) / dh_grid
        
        # Validate shapes
        if time_series.shape[::2] != xB.shape:
            raise ValueError(f"[0,2] dim of time_series {time_series.shape} != xB shape {xB.shape}")
        if time_series.shape[0] != self.n_obs:
            raise ValueError(f"time_series n_obs {time_series.shape[0]} != G n_obs {self.n_obs}")
        if B0_grid.shape[0] != self.n_obs:
            raise ValueError(f"B0_grid n_obs {B0_grid.shape[0]} != n_obs {self.n_obs}")
        if B0_grid.shape[1] != self.n_altitude:
            raise ValueError(f"B0_grid n_altitudes {B0_grid.shape[1]} != altitude_grid length {len(altitude_grid)}")
        if mlat.shape[0] != self.n_obs:
            raise ValueError(f"mlat n_obs {mlat.shape[0]} != n_obs {self.n_obs}")
        
        # Initialize model parameters
        if m0 is not None:
            if m0.size != self.n_model:
                raise ValueError(f"Initial model length {m0.size} != n_model {self.n_model}")
            self.m = m0.copy()
        else:
            self.m = np.zeros(self.n_model)
        
        # Compute scaling based on G^T G diagonal
        self.gtg_mag = np.median(np.diag(self.G.T @ self.G))
    
        # Build first-order difference matrix (for 1st order regularization)
        # Assumes model parameters are spatially ordered
        self.LL = LL
        if self.LL is not None:
            self.l2 = self.l2 * self.gtg_mag / np.median(np.diag(self.LL))
    
        self.l1 = self.l1 * self.gtg_mag
    
    def get_altitude_correction(self, dhp: float, ah: float, obs_idx: int) -> float:
        """
        Get altitude correction for specific observation based on magnetic latitude.
        
        altitude_correction = dhp + ah * (90 - |mlat|)
        
        Parameters:
        -----------
        dhp : float
            Altitude correction at pole (m)
        ah : float
            Altitude gradient (m/degree)
        obs_idx : int
            Observation index
            
        Returns:
        --------
        float : Altitude correction in meters
        """
        mlat_abs = np.abs(self.mlat[obs_idx])
        return dhp + ah * (90 - mlat_abs)
    
    def interpolate_B0(self, dh: float, obs_idx: int) -> float:
        """
        Interpolate B0 for a single observation at altitude = h_ref + dh.
        
        Parameters:
        -----------
        dh : float
            Altitude correction in m
        obs_idx : int
            Observation index
            
        Returns:
        --------
        float : Interpolated B0 value
        """
        target_altitude = self.h_ref + dh
        
        # Find bracketing indices
        idx = np.searchsorted(self.altitude_grid, target_altitude)
        
        if idx == 0:
            return self.B0_grid[obs_idx, 0]
        elif idx >= len(self.altitude_grid):
            return self.B0_grid[obs_idx, -1]
        else:
            # Linear interpolation
            alt_low = self.altitude_grid[idx-1]
            alt_high = self.altitude_grid[idx]
            w = (target_altitude - alt_low) / (alt_high - alt_low)
            return (1 - w) * self.B0_grid[obs_idx, idx-1] + w * self.B0_grid[obs_idx, idx]
    
    def interpolate_dB0_dh(self, dh: float, obs_idx: int) -> float:
        """
        Interpolate dB0/dh for a single observation at altitude = h_ref + dh.
        
        Parameters:
        -----------
        dh : float
            Altitude correction in m
        obs_idx : int
            Observation index
            
        Returns:
        --------
        float : Interpolated dB0/dh value
        """
        target_altitude = self.h_ref + dh
        
        # Find bracketing indices
        idx = np.searchsorted(self.altitude_grid[:-1], target_altitude)
        
        if idx == 0:
            return self.dB0_dh[obs_idx, 0]
        elif idx >= len(self.altitude_grid) - 1:
            return self.dB0_dh[obs_idx, -1]
        else:
            # Linear interpolation
            alt_low = self.altitude_grid[idx-1]
            alt_high = self.altitude_grid[idx]
            w = (target_altitude - alt_low) / (alt_high - alt_low)
            return (1 - w) * self.dB0_dh[obs_idx, idx-1] + w * self.dB0_dh[obs_idx, idx]
    
    def get_all_B0(self, dhp: float, ah: float) -> np.ndarray:
        """
        Get B0 for all observations with latitude-dependent altitude correction.
        
        Parameters:
        -----------
        dhp : float
            Altitude correction at pole (m)
        ah : float
            Altitude gradient (m/degree)
            
        Returns:
        --------
        np.ndarray : B0 values for all observations
        """
        B0 = np.zeros(self.n_obs)
        for obs_idx in range(self.n_obs):
            dh = self.get_altitude_correction(dhp, ah, obs_idx)
            B0[obs_idx] = self.interpolate_B0(dh, obs_idx)
        return B0
        
    def forward_model(self, m: np.ndarray) -> np.ndarray:
        """
        Forward model: compute parallel magnetic field perturbation from current system.
        
        Parameters:
        -----------
        m : np.ndarray
            Model parameters (current system)
            
        Returns:
        --------
        np.ndarray : Predicted parallel magnetic field perturbation (dB_parallel) at observation points
        """
        return self.G @ m
    
    def predict_total_field(self, dB_parallel: np.ndarray, B0: np.ndarray) -> np.ndarray:
        """
        Convert parallel magnetic field perturbation to total magnetic field.
        
        Given: dB = (B^2 - B0^2) / (2*B0)
        Solve for B: B = sqrt(dB * 2 * B0 + B0^2)
        
        Parameters:
        -----------
        dB_parallel : np.ndarray
            Parallel component of magnetic field perturbation
        B0 : np.ndarray
            Background magnetic field at current altitude
            
        Returns:
        --------
        np.ndarray : Total magnetic field
        """
        argument = dB_parallel * 2 * B0 + B0**2
        
        if np.any(argument < 0):
            print(f"Warning: Negative argument in sqrt. Min dB_parallel: {np.min(dB_parallel):.6f}")
            argument = np.maximum(argument, 0)
        
        return np.sqrt(argument)
    
    def fit_parabola_linear(self, spectrum: np.ndarray, xB_axis: np.ndarray, B_peak: float) -> Tuple[np.ndarray, float]:
        """
        Fit inverted parabola to spectrum with fixed center using linear least squares.
        
        Model: y = a₀ + a₂*(x - B_peak)²
        where a₂ < 0 for inverted parabola
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Observed spectrum values
        xB_axis : np.ndarray
            Magnetic field axis
        B_peak : float
            Fixed center location from physics model
            
        Returns:
        --------
        fitted_spectrum : np.ndarray
            Best-fit inverted parabola
        misfit : float
            Sum of squared residuals
        """
        # Shifted coordinates
        u = xB_axis - B_peak
        u_squared = u**2
        
        # Design matrix [1, u²]
        # Note: No linear term for symmetric peak
        A = np.column_stack([np.ones_like(u), u_squared])
        
        # Solve linear system: spectrum = A @ coeffs
        coeffs, residuals, _, _ = np.linalg.lstsq(A, spectrum, rcond=None)
        
        # Extract coefficients
        a0, a2 = coeffs
        
        # Generate fitted spectrum
        fitted_spectrum = a0 + a2 * u_squared
        
        # Calculate misfit
        residual = spectrum - fitted_spectrum
        misfit = np.sum(residual**2)
        
        return fitted_spectrum, misfit
    
    def fit_all_spectra(self, B_peaks: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit inverted parabolas to all spectra with centers fixed at B_peaks.
        
        Parameters:
        -----------
        B_peaks : np.ndarray
            Peak locations from physics model (n_obs,)
            
        Returns:
        --------
        fitted_spectra : np.ndarray
            Best-fit spectra (n_obs x n_spectra x n_spectral_points)
        total_misfit : float
            Total sum of squared residuals
        """
        fitted_spectra = np.zeros_like(self.time_series)
        total_misfit = 0.0
        
        for obs_idx in range(self.n_obs):
            xB_axis = self.xB[obs_idx, :]
            B_peak = B_peaks[obs_idx]
            
            for spec_idx in range(self.n_spectra):
                spectrum = self.time_series[obs_idx, spec_idx, :]
                
                # Fit inverted parabola with fixed center
                fitted_spectrum, misfit = self.fit_parabola_linear(spectrum, xB_axis, B_peak)
                
                fitted_spectra[obs_idx, spec_idx, :] = fitted_spectrum
                total_misfit += misfit
        
        return fitted_spectra, total_misfit
    
    def objective_function(self, params: np.ndarray) -> float:
        """
        Objective function for the joint inversion with Tikhonov regularization.
        
        Parameters now include both model parameters and altitude parameters.
        
        Parameters:
        -----------
        params : np.ndarray
            Combined parameters [m, dhp, ah] where m has n_model elements
            
        Returns:
        --------
        float : Objective function value
        """
        # Unpack parameters
        m = params[:-2]
        dhp = params[-2]
        ah = params[-1]
        
        # Get B0 for all observations with latitude-dependent altitude
        B0 = self.get_all_B0(dhp, ah)
        
        # Compute peak locations from physics
        dB_parallel = self.forward_model(m)
        B_peaks = self.predict_total_field(dB_parallel, B0)
        
        # Fit parabolas to all spectra with these fixed centers
        _, total_misfit = self.fit_all_spectra(B_peaks)
        
        # Tikhonov regularization on model parameters only
        # 0th order - damping (scaled)
        reg0 = self.l1 * m.T.dot(m)
        
        # 1st order - smoothing (scaled)
        if self.LL is not None:
            reg1 = self.l2 * (m.T.dot(self.LL).dot(m))
        else:
            reg1 = 0.0
        
        return total_misfit + reg0 + reg1
    
    def jacobian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute analytical Jacobian of the objective function.
        
        Now includes gradients w.r.t. model parameters and altitude parameters.
        
        Parameters:
        -----------
        params : np.ndarray
            Combined parameters [m, dhp, ah]
            
        Returns:
        --------
        np.ndarray : Gradient vector [dm, d_dhp, d_ah]
        """
        # Unpack parameters
        m = params[:-2]
        dhp = params[-2]
        ah = params[-1]
        
        # Forward model
        dB_parallel = self.forward_model(m)
        
        # Initialize gradient components
        grad_data_m = np.zeros(self.n_model)
        grad_data_dhp = 0.0
        grad_data_ah = 0.0
        
        # Compute data misfit gradient
        for obs_idx in range(self.n_obs):
            # Get altitude correction for this observation
            dh = self.get_altitude_correction(dhp, ah, obs_idx)
            B0_obs = self.interpolate_B0(dh, obs_idx)
            dB0_dh_obs = self.interpolate_dB0_dh(dh, obs_idx)
            
            # Predict total field for this observation
            B_peak = np.sqrt(dB_parallel[obs_idx] * 2 * B0_obs + B0_obs**2)
            
            xB_axis = self.xB[obs_idx, :]
            
            # Derivative of misfit w.r.t. peak location for this observation
            dJ_dB = 0.0
            
            for spec_idx in range(self.n_spectra):
                spectrum = self.time_series[obs_idx, spec_idx, :]
                
                # Current parabola fit
                u = xB_axis - B_peak
                u_squared = u**2
                A = np.column_stack([np.ones_like(u), u_squared])
                
                # Solve for coefficients
                coeffs = np.linalg.lstsq(A, spectrum, rcond=None)[0]
                a0, a2 = coeffs
                
                # Residual
                fitted = a0 + a2 * u_squared
                residual = spectrum - fitted
                
                # Contribution to dJ/dB
                dJ_dB += 4 * np.sum(residual * a2 * u)
            
            # For model parameters gradient
            if B_peak > 0:
                dB_ddB = B0_obs / B_peak
            else:
                dB_ddB = 0
            
            # Accumulate gradient w.r.t. m
            grad_data_m += dJ_dB * dB_ddB * self.G[obs_idx, :]
            
            # For altitude gradients: dJ/d(param) = dJ/dB * dB/dB0 * dB0/d(dh) * d(dh)/d(param)
            if B_peak > 0:
                dB_dB0 = (B0_obs + dB_parallel[obs_idx]) / B_peak
            else:
                dB_dB0 = 1
            
            # Common factor for altitude gradients
            altitude_factor = dJ_dB * dB_dB0 * dB0_dh_obs
            
            # Gradient w.r.t. dhp: d(dh)/d(dhp) = 1
            grad_data_dhp += altitude_factor * 1.0
            
            # Gradient w.r.t. ah: d(dh)/d(ah) = (90 - |mlat|)
            mlat_abs = np.abs(self.mlat[obs_idx])
            grad_data_ah += altitude_factor * (90 - mlat_abs)
        
        # Add regularization gradients for model parameters
        # 0th order
        grad_reg0 = 2 * self.l1 * m
                
        # 1st order
        if self.LL is not None:
            grad_reg1 = 2*self.l2*(self.LL.dot(m))
        else:
            grad_reg1 = np.zeros(self.n_model)
        
        # Combine gradients
        grad_m = grad_data_m + grad_reg0 + grad_reg1
        
        # Print diagnostics
        print(f"Gradient w.r.t. dhp: {grad_data_dhp:.3e}, ah: {grad_data_ah:.3e}")
        print(f"Mean |gradient w.r.t. m|: {np.mean(np.abs(grad_m)):.3e}")
        
        return np.concatenate([grad_m, [grad_data_dhp, grad_data_ah]])
    
    def invert(self, initial_guess: Optional[np.ndarray] = None, 
               dhp_initial: Optional[float] = None,
               ah_initial: Optional[float] = None,
               method: str = 'L-BFGS-B', max_iter: int = 100,
               dhp_bounds: Tuple[float, float] = (-35e3, 65e3),
               ah_bounds: Tuple[float, float] = (-1e3, 1e3)) -> dict:
        """
        Perform the joint inversion.
        
        Parameters:
        -----------
        initial_guess : np.ndarray, optional
            Initial guess for model parameters
        dhp_initial : float, optional
            Initial guess for altitude correction at pole (m)
        ah_initial : float, optional
            Initial guess for altitude gradient (m/degree)
        method : str
            Optimization method
        max_iter : int
            Maximum number of iterations
        dhp_bounds : tuple
            Bounds on altitude correction at pole (m)
        ah_bounds : tuple
            Bounds on altitude gradient (m/degree)
            
        Returns:
        --------
        dict : Inversion results
        """
        # Build initial parameter vector
        if initial_guess is not None:
            if len(initial_guess) != self.n_model:
                raise ValueError(f"Initial guess length {len(initial_guess)} != n_model {self.n_model}")
            m0 = initial_guess.copy()
        else:
            m0 = self.m.copy()
            
        if dhp_initial is not None:
            dhp0 = dhp_initial
        else:
            dhp0 = self.dhp
            
        if ah_initial is not None:
            ah0 = ah_initial
        else:
            ah0 = self.ah
            
        # Combined parameter vector [m, dhp, ah]
        x0 = np.concatenate([m0, [dhp0, ah0]])
        
        # Set bounds
        bounds = [(None, None)] * self.n_model  # No bounds on model parameters
        bounds.append(dhp_bounds)  # Bounds on altitude correction at pole
        bounds.append(ah_bounds)   # Bounds on altitude gradient
        
        print("Starting linear parabolic inversion with latitude-dependent altitude")
        print(f"  Model parameters: {self.n_model}")
        print(f"  Number of observations: {self.n_obs}")
        print(f"  Spectra per observation: {self.n_spectra}")
        print(f"  0th order regularization: {self.l1}")
        print(f"  1st order regularization: {self.l2}")
        print(f"  Median of GTG diagonal: {self.gtg_mag:.2e}")
        print(f"  Initial altitude at pole: {dhp0:.1f} m")
        print(f"  Initial altitude gradient: {ah0:.1f} m/degree")
        print(f"  Pole altitude bounds: [{dhp_bounds[0]:.1f}, {dhp_bounds[1]:.1f}] m")
        print(f"  Gradient bounds: [{ah_bounds[0]:.1f}, {ah_bounds[1]:.1f}] m/degree")
        
        # Print example altitudes
        mlat_examples = [90, 70, 50]
        print(f"  Example initial altitudes:")
        for mlat in mlat_examples:
            alt = self.h_ref + dhp0 + ah0 * (90 - mlat)
            print(f"    mlat={mlat}°: {alt/1e3:.1f} km")
        
        # Initial evaluation
        f0 = self.objective_function(x0)
        print(f"  Initial objective: {f0:.6f}")
        
        # Optimization
        result = opt.minimize(
            fun=self.objective_function,
            x0=x0,
            method=method,
            jac=self.jacobian if method in ['L-BFGS-B', 'BFGS'] else None,
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': True}
        )
        
        # Extract final results
        params_final = result.x
        m_final = params_final[:-2]
        dhp_final = params_final[-2]
        ah_final = params_final[-1]
        
        # Compute final fields
        B0_final = self.get_all_B0(dhp_final, ah_final)
        dB_parallel_final = self.forward_model(m_final)
        B_peaks_final = self.predict_total_field(dB_parallel_final, B0_final)
        
        # Get final fitted spectra
        fitted_spectra_final, _ = self.fit_all_spectra(B_peaks_final)
        
        # Print final altitude model
        print(f"\nFinal altitude model:")
        print(f"  Altitude at pole: {self.h_ref/1e3 + dhp_final/1e3:.1f} km (correction: {dhp_final/1e3:+.1f} km)")
        print(f"  Altitude gradient: {ah_final:.1f} m/degree")
        print(f"  Example final altitudes:")
        for mlat in mlat_examples:
            alt = self.h_ref + dhp_final + ah_final * (90 - mlat)
            print(f"    mlat={mlat}°: {alt/1e3:.1f} km")
        
        return {
            'model': m_final,
            'dhp': dhp_final,
            'ah': ah_final,
            'altitude_at_pole': self.h_ref + dhp_final,
            'altitude_gradient': ah_final,
            'B0_final': B0_final,
            'dB_parallel_modeled': dB_parallel_final,
            'B_total_modeled': B_peaks_final,
            'fitted_spectra': fitted_spectra_final,
            'optimization_result': result,
            'final_objective': result.fun
        }
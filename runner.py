#%% Import

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from secsy import get_SECS_B_G_matrices, CSgrid, CSprojection
from secsy import spherical 
from dipole import Dipole # https://github.com/klaundal/dipole
from apexpy import Apex
import ppigrf
from tqdm import tqdm
from datetime import datetime

#%%

os.chdir('/home/bing/Dropbox/work/code/repos/PZIE')
from PZIE_v6 import ZeemanLinearParabolicInversion as ZI

#%%

def get_grid(sc_lon0, sc_lat0, sc_ve0, sc_vn0, map_params):
    # set up the grid
    position = (sc_lon0, sc_lat0)
    orientation = (sc_vn0, -sc_ve0) # align coordinate system such that xi axis points right wrt to satellite velocity vector, and eta along velocity
    projection = CSprojection(position, orientation)
    L, W, LRES, WRES, wshift = map_params['L'], map_params['W'], map_params['LRES'], map_params['WRES'], map_params['wshift']
    grid = CSgrid(projection, L, W, LRES, WRES, wshift = wshift, R = map_params['RI'])# *  1e-3)
    return grid


def get_LL(grid, apx, hI):
    # set up matrix that produces gradients in the magnetic eastward direction, and use to construct regularization matrix LL:
    Le, Ln = grid.get_Le_Ln()
    f1, f2 = apx.basevectors_qd(grid.lat.flatten(), grid.lon.flatten(), hI, coords='geo')
    f1 = f1/np.linalg.norm(f1, axis = 0) # normalize
    L = Le * f1[0].reshape((-1, 1)) + Ln * f1[1].reshape((-1, 1))
    LL = L.T.dot(L)
    return LL

apx = Apex(2025, refh = 110)
OBSHEIGHT = 80e3
RE = 6371.2*1e3
d2r = np.pi / 180
RI = RE + 110e3 # radius of the ionosphere
dpl = Dipole(epoch = 2025) # initialize Dipole object

#%%

base = '/home/bing/Dropbox/work/temp_storage/EZIE/'
folders = os.listdir(base + 'data2_restored')

for folder in tqdm(folders, total=len(folders)):
    MEM1 = pd.read_pickle(f'{base}data2_restored/{folder}/MEM_1.pkl')
    MEM2 = pd.read_pickle(f'{base}data2_restored/{folder}/MEM_2.pkl')
    MEM3 = pd.read_pickle(f'{base}data2_restored/{folder}/MEM_3.pkl')
    MEM4 = pd.read_pickle(f'{base}data2_restored/{folder}/MEM_4.pkl')

#%% Timespan and satellite velocity

    # calculate SC velocity
    te, tn = spherical.tangent_vector(MEM1['lat'][:-1], MEM1['lon'][:-1],
                                      MEM1['lat'][1 :], MEM1['lon'][1: ])

    ve = np.hstack((te, np.nan))
    vn = np.hstack((tn, np.nan))

    # get index of central point of analysis interval:
    tm = te.size//2

    # spacecraft velocity at central time:
    v = np.array((ve[tm], vn[tm]))

    # spacecraft lat and lon at central time:
    sc_lat0 = MEM1['lat'][tm]
    sc_lon0 = MEM1['lon'][tm]

    # limits of analysis interval:
    t0 = 0
    t1 = te.size - 1

    # get unit vectors pointing at satellite (Cartesian vectors)
    rs = []
    for t in [t0, tm, t1]:
        rs.append(np.array([np.cos(MEM1['lat'][t] * d2r) * np.cos(MEM1['lon'][t] * d2r),
                            np.cos(MEM1['lat'][t] * d2r) * np.sin(MEM1['lon'][t] * d2r),
                            np.sin(MEM1['lat'][t] * d2r)]))

#%% Define map paramters

    # dimensions of analysis region/d (in km)
    W = 2000000 + RI * np.arccos(np.sum(rs[0]*rs[-1])) # km

    map_params = {'LRES':40.*1e3,
                  'WRES':40.*1e3,
                  'W': W, # along-track dimension of analysis grid (TODO: This is more a proxy than a precise description)
                  'L': 2000*1e3, # cross-track dimension of analysis grid (TODO: Same as above)
                  'wshift':25, # shift the grid center wres km in cross-track direction
                  'total_time_window':6*60,
                  'strip_time_window':30,
                  'RI':RI, # height of the ionosphere [m]
                  'RE':RE,
                  'Rez':RE+OBSHEIGHT
                  }

#%% Grab data from selected time

    obs = {'lat': [], 'lon': [],
           'lat_1': [], 'lat_2': [], 'lat_3': [], 'lat_4': [], 
           'lon_1': [], 'lon_2': [], 'lon_3': [], 'lon_4': []}
    for i, MEM in enumerate([MEM1, MEM2, MEM3, MEM4]):
    
        obs['lat'] += list(MEM['lat'])
        obs['lon'] += list(MEM['lon'])
    
        # for plotting tracks
        obs[f'lat_{i+1}'] = list(MEM['lat'])
        obs[f'lon_{i+1}'] = list(MEM['lon'])

    for key in obs.keys():
        obs[key] = np.array(obs[key])

    obs['x'] = np.vstack((MEM1['x'], MEM2['x'], MEM3['x'], MEM4['x']))
    obs['x'] *= 1e-9
    obs['peaks'] = np.vstack((MEM1['peaks'][:, 2:4, :], MEM2['peaks'][:, 2:4, :], MEM3['peaks'][:, 2:4, :], MEM4['peaks'][:, 2:4, :]))
    
#%% Define grid

    grid = get_grid(sc_lon0, sc_lat0, v[0], v[1], map_params)

    date = MEM1['time'][0]

#%% Calculate main magnetic field
    
    altitudes = np.arange(50, 150.1, 0.1)*1e3
    B0_grid = np.zeros((obs['x'].shape[0], altitudes.size))
    for i in tqdm(range(altitudes.size), total=altitudes.size):
        Be0, Bn0, Bu0 = map(np.ravel, ppigrf.igrf(obs['lon'], obs['lat'], altitudes[i]*1e-3, datetime(2025,1,1)))
        B0_vector = np.vstack((Be0, Bn0, Bu0))
        B0 = np.linalg.norm(B0_vector, axis = 0)
        B0_grid[:, i] = B0
    B0_grid *= 1e-9

    Be0, Bn0, Bu0 = map(np.ravel, ppigrf.igrf(obs['lon'], obs['lat'], (OBSHEIGHT)*1e-3, date))
    B0_vector = np.vstack((Be0, Bn0, Bu0))
    b0 = B0_vector / np.linalg.norm(B0_vector, axis = 0)

#%% Prep matrices
    
    GBe, GBn, GBu = get_SECS_B_G_matrices(obs['lat'], obs['lon'], (OBSHEIGHT) + RE, grid.lat, grid.lon, RI = RI)
    G = GBe * b0[0].reshape((-1, 1)) + GBn * b0[1].reshape((-1, 1)) + GBu * b0[2].reshape((-1, 1))

    LL = get_LL(grid, apx, (RI-RE)*1e-3)

#%% mlat

    dpl = Dipole(epoch = 2025)
    mlat, _ = dpl.geo2mag(obs['lat'], obs['lon'])

#%% Init
    
    m0 = np.zeros(G.shape[1])

#%% Run inversion

    l1 = 1e17
    l2 = 1e21
    inversion = ZI(G, B0_grid, altitudes, obs['peaks'], obs['x'], mlat, 
                   m0=m0, l1=l1, l2=l2, LL=LL)

    results = inversion.invert(max_iter=100, dhp_bounds=(-80e3, 100e3), ah_bounds=(-10e0, 10e0))

#%%

    GBegrid, GBngrid, GBugrid = get_SECS_B_G_matrices(grid.lat_mesh, grid.lon_mesh, OBSHEIGHT + RE, grid.lat, grid.lon, RI = RI)
    Brd = GBugrid.dot(results['model']).reshape(grid.xi_mesh.shape)*1e9
    Bnd = GBngrid.dot(results['model']).reshape(grid.xi_mesh.shape)*1e9
    Bed = GBegrid.dot(results['model']).reshape(grid.xi_mesh.shape)*1e9
    vmax = np.max(abs(Brd))
    clvls = np.linspace(-vmax, vmax, 40)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].contourf(grid.xi_mesh, grid.eta_mesh, Bed, cmap='bwr', levels=clvls)
    axs[1].contourf(grid.xi_mesh, grid.eta_mesh, Bnd, cmap='bwr', levels=clvls)
    axs[2].contourf(grid.xi_mesh, grid.eta_mesh, Brd, cmap='bwr', levels=clvls)
    
    for ax in axs:
        for i in range(4):
            xi, eta = grid.projection.geo2cube(obs[f'lon_{i+1}'], obs[f'lat_{i+1}'])
            ax.plot(xi, eta, linewidth=3)
    
    plt.suptitle(vmax)

#%%
    
    fig, axs = plt.subplots(2,4, figsize=(20, 10))
    for i, ax in enumerate(axs.flatten()):
        sid = 300+i
        
        ax.plot(obs['x'][sid]*1e9, obs['peaks'][sid, 0], label='Spectra \#1')
        spec = results['fitted_spectra'][sid, 0]
        ax.plot(obs['x'][sid]*1e9, spec, label='Spectral fit \#1')
        ax.plot(obs['x'][sid, np.argmin(spec)]*1e9, spec[np.argmin(spec)], '*', label='Optimized peak')
        
        ax.plot(obs['x'][sid]*1e9, obs['peaks'][sid, 1], '--', color='tab:blue')
        spec = results['fitted_spectra'][sid, 1]
        ax.plot(obs['x'][sid]*1e9, spec, '--', color='tab:orange')
        ax.plot(obs['x'][sid, np.argmin(spec)]*1e9, spec[np.argmin(spec)], '*', color='tab:green')
        
        ax.vlines(obs['x'][sid, obs['x'].shape[1]//2]*1e9, ax.get_ylim()[0], ax.get_ylim()[1], 
                  color='tab:red', label='Initial peak, 85 km')
    
    for ax in axs[-1, :]:
        ax.set_xlabel('Total B')
    for ax in axs[:, 0]:
        ax.set_ylabel('Spectral mystery units')
        
    axs[0,0].legend()


#%% Spectral movie

    def plot_peaks(ax, X, Y, B0peaks, Bpeaks, lim):
        xxB, yyB = [], []
        xxB0, yyB0 = [], []
        xxc, yyc = [], []
        xmin, xmax, ymin, ymax = lim
        for i in range(X.shape[0]):
            x = X[i, :]
            spec = Y[i, :]+i*3
            
            if xmin > np.min(x):
                xmin = np.min(x)
            if xmax < np.max(x):
                xmax = np.max(x)
            if ymin > np.min(spec):
                ymin = np.min(spec)
            if ymax < np.max(spec):
                ymax = np.max(spec)
            
            c_id = len(x)//2
            s_id = np.argmin(spec)
            B0_id = np.argmin(abs(x-B0peaks[i]))
            B_id = np.argmin(abs(x-Bpeaks[i]))
            
            if i == 0:
                ax.plot(x, spec, color='k', linewidth=.2, label='spectra')
            else:
                ax.plot(x, spec, color='k', linewidth=.2)
            
            if i == 0:
                ax.plot(x[s_id], spec[s_id], '.', color='tab:red', markersize=5, label='minimum')
            else:
                ax.plot(x[s_id], spec[s_id], '.', color='tab:red', markersize=5)
            
            xxB.append(x[B_id])
            yyB.append(spec[B_id])            
            xxB0.append(x[B0_id])
            yyB0.append(spec[B0_id])
            xxc.append(x[c_id])
            yyc.append(spec[c_id])
            
        ax.plot(xxc, yyc, color='tab:orange', linewidth=2, label='IGRF, 80 km')
        ax.plot(xxB0, yyB0, color='tab:green', linewidth=3, label='modelled B0')
        ax.plot(xxB, yyB, color='k', linewidth=.5, label='modelled B')
        return xmin, xmax, ymin, ymax

    fig, axs = plt.subplots(2, 4, figsize=(15, 15), sharex=True, sharey=True)
    lim = (1e9, 0, 1e9, 0)
    for i in range(4):        
        x = obs['x'][i*250:(i+1)*250]*1e9
        y1 = obs['peaks'][i*250:(i+1)*250, 0, :]
        y2 = obs['peaks'][i*250:(i+1)*250, 1, :]
        Bpeaks = results['B_total_modeled'][i*250:(i+1)*250]*1e9
        B0peaks = results['B0_final'][i*250:(i+1)*250]*1e9
        lim = plot_peaks(axs[0, i], x, y1, B0peaks, Bpeaks, lim)
        lim = plot_peaks(axs[1, i], x, y2, B0peaks, Bpeaks, lim)
        axs[0, i].set_title(f'MEM {i+1}, spec \#1')
        axs[1, i].set_title(f'MEM {i+1}, spec \#2')
        axs[1, i].set_xlabel('B total [nT]')
    axs[0,-1].legend(bbox_to_anchor=(1.05, 1.05))
    xmin, xmax, ymin, ymax = lim
    axs[0,0].set_xlim((xmin, xmax))
    axs[0,0].set_ylim((ymin, ymax))
    axs[0,0].set_ylabel('Spectra')
    axs[1,0].set_ylabel('Spectra')
    plt.suptitle(f'{results["optimization_result"]["nit"]} iterations of PZIE, dhp {results["dhp"]*1e-3:.1f} km, ah {results["ah"]*1e-3:.1f} km/mlat', y=.95)
        

#%% Spectral movie latitude edition

    def plot_peaks(ax, X, Y, B0peaks, Bpeaks, lim, mlat):
        xxB, yyB = [], []
        xxB0, yyB0 = [], []
        xxc, yyc = [], []
        xmin, xmax, ymin, ymax = lim
        scale = 1
        for i in range(X.shape[0]):
            x = X[i, :]
            spec = Y[i, :]
            spec -= (spec[0] + spec[-1])/2
            spec /= np.std(spec)
            spec += mlat[i]*scale
            
            if xmin > np.min(x):
                xmin = np.min(x)
            if xmax < np.max(x):
                xmax = np.max(x)
            if ymin > np.min(spec):
                ymin = np.min(spec)
            if ymax < np.max(spec):
                ymax = np.max(spec)
            
            c_id = len(x)//2
            s_id = np.argmin(spec)
            B0_id = np.argmin(abs(x-B0peaks[i]))
            B_id = np.argmin(abs(x-Bpeaks[i]))
            
            if i == 0:
                ax.plot(x, spec, color='k', linewidth=.2, label='spectra')
            else:
                ax.plot(x, spec, color='k', linewidth=.2)
            
            if i == 0:
                ax.plot(x[s_id], spec[s_id], '.', color='tab:red', markersize=5, label='minimum')
            else:
                ax.plot(x[s_id], spec[s_id], '.', color='tab:red', markersize=5)
            
            xxB.append(x[B_id])
            yyB.append(spec[B_id])            
            xxB0.append(x[B0_id])
            yyB0.append(spec[B0_id])
            xxc.append(x[c_id])
            yyc.append(spec[c_id])
            
        ax.plot(xxc, yyc, color='tab:orange', linewidth=2, label='IGRF, 80 km')
        ax.plot(xxB0, yyB0, color='tab:green', linewidth=3, label='modelled B0')
        ax.plot(xxB, yyB, color='k', linewidth=.5, label='modelled B')
        return xmin, xmax, ymin, ymax

    fig, axs = plt.subplots(2, 4, figsize=(15, 15), sharex=True, sharey=True)
    lim = (1e9, 0, 1e9, 0)
    dpl = Dipole(epoch = 2025)
    for i in range(4):        
        x = obs['x'][i*250:(i+1)*250]*1e9
        y1 = obs['peaks'][i*250:(i+1)*250, 0, :]
        y2 = obs['peaks'][i*250:(i+1)*250, 1, :]
        mlati = mlat[i*250:(i+1)*250]
        Bpeaks = results['B_total_modeled'][i*250:(i+1)*250]*1e9
        B0peaks = results['B0_final'][i*250:(i+1)*250]*1e9
        lim = plot_peaks(axs[0, i], x, y1, B0peaks, Bpeaks, lim, mlati)
        lim = plot_peaks(axs[1, i], x, y2, B0peaks, Bpeaks, lim, mlati)
        axs[0, i].set_title(f'MEM {i+1}, spec \#1')
        axs[1, i].set_title(f'MEM {i+1}, spec \#2')
        axs[1, i].set_xlabel('B total [nT]')
    axs[0,-1].legend(bbox_to_anchor=(1.05, 1.05))
    xmin, xmax, ymin, ymax = lim
    axs[0,0].set_xlim((xmin, xmax))
    axs[0,0].set_ylim((ymin, ymax))
    axs[0,0].set_ylabel('Magnetic lat')
    axs[1,0].set_ylabel('Magnetic lat')
    plt.suptitle(f'{results["optimization_result"]["nit"]} iterations of PZIE, dhp {results["dhp"]*1e-3:.1f} km, ah {results["ah"]*1e-3:.1f} km/mlat', y=.95)
    
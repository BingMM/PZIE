#%% Import

from netCDF4 import Dataset
import ppigrf
from datetime import datetime, timedelta
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from scipy.signal import correlate
import pickle

plt.ioff()

#%% Func

def extract_non_nan_sequences(arr, arr2, matrix):
    """
    Extract contiguous non-NaN sequences from a 1D array and apply the same
    slices to a corresponding 2D array.

    Parameters:
        arr (1D np.ndarray): Array with possible NaNs (length N)
        matrix (2D np.ndarray): Shape (N, M), same first dimension as arr

    Returns:
        arr_chunks (list of 1D arrays): Non-NaN chunks from arr
        matrix_chunks (list of 2D arrays): Corresponding rows from matrix
    """
    idx = np.flatnonzero(~np.isnan(arr))
    if len(idx) == 0:
        return [], []

    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)

    arr_chunks = [arr[g] for g in groups]
    arr2_chunks = [arr2[g] for g in groups]
    matrix_chunks = [matrix[g, :] for g in groups]

    return arr_chunks, arr2_chunks, matrix_chunks

def find_symmetry_center(line):
    """
    Find the center of symmetry in a 1D W-shaped signal.

    Parameters:
        line (np.ndarray): 1D input array (e.g., W-shaped)

    Returns:
        center (float): Estimated center index of symmetry
    """
    line = np.asarray(line)
    flipped = line[::-1]

    # Compute normalized cross-correlation
    corr = correlate(line - np.mean(line), flipped - np.mean(flipped), mode='full')
    lags = np.arange(-len(line) + 1, len(line))

    best_lag = lags[np.argmax(abs(corr))]
    center = (len(line) - 1) / 2 + best_lag / 2  # Midpoint shift

    return int(center)

def upsample_spline(t, y, t_new, smooth_factor=0.5):
    # Mask out NaNs
    mask = ~np.isnan(y)
    t_valid = t[mask]
    y_valid = y[mask]

    # Fit spline only to valid data
    spline = UnivariateSpline(t_valid, y_valid, s=smooth_factor)

    # Evaluate spline on new time base
    return spline(t_new)

#%% Data path

base = '/home/bing/Downloads/06172025_apl_currentruns/'
#fn = base + 'ezie_l1_20250504_005253_sva_v000_r000.nc4'
#fn = base + 'ezie_l1_20250505_023909_sva_v000_r000.nc4'
fn = base + 'ezie_l1_20250507_011803_sva_v000_r000.nc4'

#%% Load data

MEM = 4

ds = Dataset(fn)

data = ds['CalibratedSceneTemperatures']['ta' + str(MEM)][:].filled(np.nan)
lon = ds['Geolocation']['obs_lon' + str(MEM)][:].filled(np.nan)
lat = ds['Geolocation']['obs_lat' + str(MEM)][:].filled(np.nan)

kHz = ds['FrequencyGrid']['frequency'][:].filled(np.nan)*1e6 - 118728e3
KHz = kHz[300:650]
dkHz = np.median(np.diff(kHz[0]))

time = ds['Time']['time_tai'][:].filled(np.nan)

if MEM == 4:
    dat1 = data[:, 300:650, 0]
    dat0 = data[:, 300:650, 1]
    dat2 = data[:, 300:650, 2]
elif MEM == 2:
    dat0 = data[:, 300:650, 0]
    dat1 = data[:, 300:650, 1]
    dat2 = data[:, 300:650, 3]
else:
    dat0 = data[:, 300:650, 0]
    dat1 = data[:, 300:650, 1]
    dat2 = data[:, 300:650, 2]

datc = dat1 + dat0

#%% Divide into orbits

_,   time, _ = extract_non_nan_sequences(lat, time, dat0)
_,   _,    dat0 = extract_non_nan_sequences(lat, lon, dat0)
_,   _,    dat1 = extract_non_nan_sequences(lat, lon, dat1)
_,   _,    dat2 = extract_non_nan_sequences(lat, lon, dat2)
lat, lon,  datc = extract_non_nan_sequences(lat, lon, datc)

o = 1

date0 = datetime(2000, 1, 1, 0, 0)
for i in range(len(time)):
    time[i] = np.array([date0 + timedelta(seconds=s) for s in time[i]])

#%% Plot examples

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i in range(5):
    axs[0,0].plot(dat0[o][i*10])
    axs[0,1].plot(dat1[o][i*10])
    axs[1,0].plot(dat2[o][i*10])
    axs[1,1].plot(datc[o][i*10])

#%% Discard start and end

time = [t[10:-11] for t in time]
lat = [l[10:-11] for l in lat]
lon = [l[10:-11] for l in lon]
dat0 = [d[10:-11, :] for d in dat0]
dat1 = [d[10:-11, :] for d in dat1]
dat2 = [d[10:-11, :] for d in dat2]
datc = [d[10:-11, :] for d in datc]

#%% Plot it

fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
axs[0,0].imshow(dat0[o])
axs[0,1].imshow(dat1[o])
axs[1,0].imshow(dat2[o])
axs[1,1].imshow(datc[o])

#%% Upsample

uf = 100 # upsample factor

w = dat0[o].shape[1]
x = np.arange(w)

x_new = np.linspace(x[0], x[-1], w*uf)

cut_end = 50*uf

dat0_, dat1_, dat2_, datc_  = [], [], [], []
for (d0, d1, d2, dc) in tqdm(zip(dat0, dat1, dat2, datc), total=len(dat0)):
    n = d0.shape[0]
    d0_, d1_ = np.zeros((n, x_new.size-2*cut_end)), np.zeros((n, x_new.size-2*cut_end))
    d2_, dc_ = np.zeros((n, x_new.size-2*cut_end)), np.zeros((n, x_new.size-2*cut_end))
    for i in range(n):
        d0_[i] = upsample_spline(x, d0[i], x_new, smooth_factor=.5)[cut_end:-cut_end]
        d1_[i] = upsample_spline(x, d1[i], x_new, smooth_factor=.5)[cut_end:-cut_end]
        d2_[i] = upsample_spline(x, d2[i], x_new, smooth_factor=.5)[cut_end:-cut_end]
        dc_[i] = upsample_spline(x, dc[i], x_new, smooth_factor=.5)[cut_end:-cut_end]
    dat0_.append(d0_)
    dat1_.append(d1_)
    dat2_.append(d2_)
    datc_.append(dc_)

#%% Plot upsampled data

X, Y = np.meshgrid(x, np.arange(dat0[o].shape[0]))
X_new, Y_new = np.meshgrid(x_new[cut_end:-cut_end], np.arange(dat0[o].shape[0]))

fig, axs = plt.subplots(2, 4, figsize=(10, 5), sharex=True)
axs[0,0].contourf(X, Y, dat0[o])
axs[0,1].contourf(X, Y, dat1[o])
axs[0,2].contourf(X, Y, dat2[o])
axs[0,3].contourf(X, Y, datc[o])
axs[1,0].contourf(X_new, Y_new, dat0_[o])
axs[1,1].contourf(X_new, Y_new, dat1_[o])
axs[1,2].contourf(X_new, Y_new, dat2_[o])
axs[1,3].contourf(X_new, Y_new, datc_[o])

#%% Plot upsampled data examples

i = 50
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0,0].plot(x, dat0[o][i])
axs[0,1].plot(x, dat1[o][i])
axs[1,0].plot(x, dat2[o][i])
axs[1,1].plot(x, datc[o][i])

axs[0,0].plot(x_new[cut_end:-cut_end], dat0_[o][i])
axs[0,1].plot(x_new[cut_end:-cut_end], dat1_[o][i])
axs[1,0].plot(x_new[cut_end:-cut_end], dat2_[o][i])
axs[1,1].plot(x_new[cut_end:-cut_end], datc_[o][i])

#%% Replace data with upsampled version

dat0, dat1, dat2, datc = dat0_, dat1_, dat2_, datc_
x = np.arange(dat0[o].shape[1])
X, Y = np.meshgrid(x, np.arange(dat0[o].shape[0]))

dkHz /= uf

#%% Find center

c_dat0, c_dat1, c_dat2, c_datc = [], [], [], []

for (d0, d1, d2, dc) in tqdm(zip(dat0, dat1, dat2, datc), total=len(dat0)):
    c_d0, c_d1, c_d2, c_dc = [], [], [], []
    for i in range(d0.shape[0]):
        c_d0.append(find_symmetry_center(d0[i, :]))
        c_d1.append(find_symmetry_center(d1[i, :]))
        c_d2.append(find_symmetry_center(d2[i, :]))
        c_dc.append(find_symmetry_center(dc[i, :]))
    c_dat0.append(np.array(c_d0).astype(int))
    c_dat1.append(np.array(c_d1).astype(int))
    c_dat2.append(np.array(c_d2).astype(int))
    c_datc.append(np.array(c_dc).astype(int))

#%% Update plot

fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
axs[0,0].contourf(X, Y, dat0[o])
axs[0,1].contourf(X, Y, dat1[o])
axs[1,0].contourf(X, Y, dat2[o])
axs[1,1].contourf(X, Y, datc[o])

axs[0,0].plot(np.array(c_dat0[o]), np.arange(len(c_dat0[o])), '.', color='tab:red')
axs[0,1].plot(np.array(c_dat1[o]), np.arange(len(c_dat1[o])), '.', color='tab:red')
axs[1,0].plot(np.array(c_dat2[o]), np.arange(len(c_dat2[o])), '.', color='tab:red')
axs[1,1].plot(np.array(c_datc[o]), np.arange(len(c_datc[o])), '.', color='tab:red')

plt.figure()
plt.plot(c_dat0[o], '.')
plt.plot(c_dat1[o], '.')
plt.plot(c_dat2[o], '.')
plt.plot(c_datc[o], '.')

#%% Center all data

cr = 80 # cut radius
dat0_, dat1_, dat2_, datc_ = [], [], [], []
for (d0, d1, d2, dc, cd) in tqdm(zip(dat0, dat1, dat2, datc, c_datc), total=len(dat0)):
    n = d0.shape[0]
    d0_, d1_, d2_, dc_ = [], [], [], []
    for i in range(n):
        # I think this is wrong
        #d0_.append(d0[i, cd[i]-cr*uf:cd[i]+(cr+1)*uf])
        #d1_.append(d1[i, cd[i]-cr*uf:cd[i]+(cr+1)*uf])
        #d2_.append(d2[i, cd[i]-cr*uf:cd[i]+(cr+1)*uf])
        #dc_.append(dc[i, cd[i]-cr*uf:cd[i]+(cr+1)*uf])
        # Fixed version
        d0_.append(d0[i, cd[i]-cr*uf:cd[i]+(cr*uf)+1])
        d1_.append(d1[i, cd[i]-cr*uf:cd[i]+(cr*uf)+1])
        d2_.append(d2[i, cd[i]-cr*uf:cd[i]+(cr*uf)+1])
        dc_.append(dc[i, cd[i]-cr*uf:cd[i]+(cr*uf)+1])
    dat0_.append(np.array(d0_))
    dat1_.append(np.array(d1_))
    dat2_.append(np.array(d2_))
    datc_.append(np.array(dc_))

#%% Update plot

x = np.linspace(-cr*uf, cr*uf, dat0_[0].shape[1])

X, Y = np.meshgrid(x, np.arange(dat0[o].shape[0]))

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0,0].contourf(X, Y, dat0_[o])
axs[0,1].contourf(X, Y, dat1_[o])
axs[1,0].contourf(X, Y, dat2_[o])
axs[1,1].contourf(X, Y, datc_[o])
for ax in axs.flatten():
    ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[1], color='tab:red')

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i in range(5):
    axs[0,0].plot(x, dat0_[o][i*10])
    axs[0,1].plot(x, dat1_[o][i*10])
    axs[1,0].plot(x, dat2_[o][i*10])
    axs[1,1].plot(x, datc_[o][i*10])
for ax in axs.flatten():
    ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[1], color='tab:red')

#%% Replace data with centered version

dat0, dat1, dat2, datc = dat0_, dat1_, dat2_, datc_

#%% Calcualte IGRF guess

date = datetime(2025, 5, 4, 1)

x0id = np.argmin(abs(x))

Be, Bn, Bu, B = [], [], [], []
split = []
delta1_B = []
delta2_B = []
for (lo, la) in tqdm(zip(lon, lat), total=len(lon)):
    Bei, Bni, Bui = ppigrf.igrf(lo, la, 80, date)
    Be.append(Bei.flatten())
    Bn.append(Bni.flatten())
    Bu.append(Bui.flatten())
    Bi = np.sqrt(Bei**2 + Bni**2 + Bui**2).flatten()
    B.append(Bi)
    split.append(Bi * 0.014012 * 2 / dkHz) # nT to kHz, to full width in kHz, to index unit
    delta1_B.append(x0id - (Bi*2*0.014012/dkHz/2).astype(int))
    delta2_B.append(x0id + (Bi*2*0.014012/dkHz/2).astype(int))

#%% Grab poly data

w = 700

p_dat, x_dat = [], []
for (d0, d1, d2, cd, del1, del2) in tqdm(zip(dat0, dat1, dat2, c_datc, delta1_B, delta2_B), total=len(dat0)):
    n = d1.shape[0]
    p_dat_ = np.zeros((n, 6, 2*w+1))
    x_dat_ = np.zeros((n, 2*w+1))
    for i in range(n):
        pid1 = del1[i]
        pid2 = del2[i]
        
        p_dat_[i, 0, :] = np.flip(d0[i, pid1-w:pid1+w+1])
        p_dat_[i, 1, :] = d0[i, pid2-w:pid2+w+1]
        
        p_dat_[i, 2, :] = np.flip(d1[i, pid1-w:pid1+w+1])
        p_dat_[i, 3, :] = d1[i, pid2-w:pid2+w+1]
        
        p_dat_[i, 4, :] = np.flip(d2[i, pid1-w:pid1+w+1])
        p_dat_[i, 5, :] = d2[i, pid2-w:pid2+w+1]
        
        x_dat_[i] = (np.arange(pid2-w, pid2+w+1) - x0id) / 0.014012 *dkHz
    
    p_dat.append(p_dat_)
    x_dat.append(x_dat_)

#%% Visualize

i = 50
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0,0].plot(x, dat0[o][i])
axs[0,1].plot(x, dat1[o][i])
axs[1,0].plot(x, dat2[o][i])
axs[1,1].plot(x, datc[o][i])

for ax in axs.flatten():
    ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[1], color='tab:red')

pid1 = delta1_B[o][i]
pid2 = delta2_B[o][i]
ax = axs[0,0]
ax.plot(x[pid1-w:pid1+w+1], np.flip(p_dat[o][i, 0, :]))
ax.plot(x[pid2-w:pid2+w+1], p_dat[o][i, 1, :])
ax.plot(x[pid1], dat0[o][i, pid1], '*', color='tab:red')
ax.plot(x[pid2], dat0[o][i, pid2], '*', color='tab:red')

ax = axs[0,1]
ax.plot(x[pid1-w:pid1+w+1], np.flip(p_dat[o][i, 2, :]))
ax.plot(x[pid2-w:pid2+w+1], p_dat[o][i, 3, :])
ax.plot(x[pid1], dat1[o][i, pid1], '*', color='tab:red')
ax.plot(x[pid2], dat1[o][i, pid2], '*', color='tab:red')

ax = axs[1,0]
ax.plot(x[pid1-w:pid1+w+1], np.flip(p_dat[o][i, 4, :]))
ax.plot(x[pid2-w:pid2+w+1], p_dat[o][i, 5, :])
ax.plot(x[pid1], dat2[o][i, pid1], '*', color='tab:red')
ax.plot(x[pid2], dat2[o][i, pid2], '*', color='tab:red')

ax = axs[1,1]
ax.plot(x[pid1], datc[o][i, pid1], '*', color='tab:red')
ax.plot(x[pid2], datc[o][i, pid2], '*', color='tab:red')

#%% Save peak data

for i in range(len(time)):
    output = {'time': time[i],
              'x': x_dat[i],
              'peaks': p_dat[i],
              'lat': lat[i],
              'lon': lon[i]
              }

    t0 = time[i][0].strftime('%Y_%m_%d_%H_%M')
    path_out = f'/home/bing/Dropbox/work/temp_storage/EZIE/data2_corrected/{t0}'
    os.makedirs(path_out, exist_ok=True)
    fn_out = path_out + f'/MEM_{MEM}.pkl'
    with open(fn_out, 'wb') as f:
        pickle.dump(output, f)


import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os 
# get F_sgs
cp = 1004
lv = 2.5104e6
lf = 0.3336e6
Rv = 461
Rd = 287
epsilon = Rd / Rv
eps0 = 29 / 18 -1  # dry_air/H2O - 1
sec_day = 86400  # seconds in a day

tstp = -1
sst_list = np.arange(280,330,10)
sgs2 = []
smooth_x = 11
smooth_t = 5*24
smx_bd = int((smooth_x - 1) / 2)

for sst in sst_list:
    fn1d = '/glade/scratch/linyao/SAMOUTPUT/csa2dLD'+str(sst)+'Kfine/dx2km8192x1xfiner/exe.sam/OUT_STAT/CSALD'+str(sst)+'Kfine_dx2km8192x1x128x3600_1dfile.nc'
    fn2d = '/glade/scratch/linyao/SAMOUTPUT/csa2dLD'+str(sst)+'Kfine/dx2km8192x1xfiner/exe.sam/OUT_2D/CSALD'+str(sst)+'Kfine_dx2km8192x1x128x3600_2dfile.nc'
    fn3d = '/glade/scratch/linyao/SAMOUTPUT/csa2dLD'+str(sst)+'Kfine/dx2km8192x1xfiner/exe.sam/OUT_3D/CSALD'+str(sst)+'Kfine_dx2km8192x1x128x3600_3dfile.nc'

    ds = xr.open_dataset(fn3d)
    t = ds['time'][:tstp].values
    x = ds['x'][0::4].values
    z = ds['z'].values
    kx = len(x)
    kz = len(z)
    kt = len(t)
    L = np.max(x)
    dt = (t[kt-1] - t[0]) * 3600 * 24 / (kt - 1)
    dx = x[1] - x[0]

    ds1 = xr.open_dataset(fn1d)
    rho = ds1['RHO'][:tstp,:].values
    rho0 = rho.reshape(kt,kz,1)

    print(np.shape(rho0))

    # SGS convert to J/m3/s
    Qt_SGS = ds['FQT'][:tstp,:,::4].values
    Qp_SGS = ds['FQP'][:tstp,:,::4].values
    Qh_SGS = ds['Ft'][:tstp,:,::4].values * cp / sec_day * rho0
    Q_SGS = Qt_SGS + Qp_SGS + Qh_SGS

    ds2 = xr.open_dataset(fn2d)
    sfc = ds2['SHF'][:tstp,::4].values + ds2['LHF'][:tstp,::4].values

    F = np.zeros((kt,kz,kx))

    zi = np.zeros((1,kz+1,1)) 
    zi[0,1:-1,0] = (z[1:]+z[:-1])/2

    dzi = zi[:,1:,:] - zi[:,:-1,:]

    dz = z[1:] - z[:-1]
    dz = np.reshape(dz, (1,kz-1,1))

    for k in np.arange(1,kz):
        F[:,k,:] = np.sum(Q_SGS[:,:k,:]*dzi[:,:k,:], axis=1)

    del Q_SGS

    Fsgs = - F + np.reshape(sfc,(kt,1,kx))
    Fsgsp = Fsgs - np.mean(Fsgs,axis=2,keepdims=True)
    del Fsgs
    del F
    
    
    y = xr.DataArray(
        data = Fsgsp,
        dims=['time','z','x'],
        coords=dict(
            time = t,
            z = z,
            x = x,
        ),
    )
    
    # smooth in time and x
    tmp = y.rolling(time=smooth_t, min_periods=int(smooth_t/2),center=True).mean().dropna("time")  # smooth in time

    print(np.shape(tmp))
    del y
    hfp1 = xr.concat([tmp[:,:,-smx_bd::], tmp, tmp[:,:,:smx_bd]], dim="x")
    del tmp
    Fsgsp = hfp1.rolling(x=smooth_x, center=True).mean("x").dropna("x")  # smooth in x 
    del hfp1
    
    
    fnh = 'MSE_sst'+str(sst)+'K.nc'
    dsh = xr.open_dataset(fnh)

    hf = dsh['hf'][:tstp,:,:].values 
    dhdz = np.zeros((kt,kz,kx))
    dhdz[:,1:,:] = (hf[:,1:,:] - hf[:,:-1,:]) / dz
    dhdz[:,0,:] = dhdz[:,1,:]

    dhdzp = dhdz - np.mean(dhdz, axis=2,keepdims=True)
    del dhdz

    y = xr.DataArray(
        data = dhdzp,
        dims=['time','z','x'],
        coords=dict(
            time = t,
            z = z,
            x = x,
        ),
    )
    
    # smooth in time and x
    tmp = y.rolling(time=smooth_t, min_periods=int(smooth_t/2),center=True).mean().dropna("time")  # smooth in time

    print(np.shape(tmp))
    del y
    hfp1 = xr.concat([tmp[:,:,-smx_bd::], tmp, tmp[:,:,:smx_bd]], dim="x")
    del tmp
    dhdzp = hfp1.rolling(x=smooth_x, center=True).mean("x").dropna("x")  # smooth in x 
    del hfp1

    sgs2.append(np.mean(Fsgsp.values * dhdzp.values, axis=2))
    del Fsgsp
    del dhdzp
    
sgs = np.asarray(sgs2)

y=xr.DataArray(
    data = sgs,
    dims=['sst','time','z'],
    coords=dict(
        sst = sst_list,
        time = t,
        z = z,
    ),
)

ds = y.to_dataset(name='SGS2')
ds.to_netcdf('SGS2.nc')


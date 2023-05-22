import numpy as np 
import xarray as xr 
import os 

# set universal parameters
cp = 1004
lv = 2.5104e6
lf = 0.3336e6
Rv = 461
Rd = 287
g  = 9.8
epsilon = Rd / Rv
eps0 = 29 / 18 -1  # dry_air/H2O - 1
sec_day = 86400  # seconds in a day

# set variables
sst = int(float(os.environ["sst"]))

fn1d = '/glade/scratch/linyao/SAMOUTPUT/csa2dLD'+str(sst)+'Kfine/dx2km8192x1xfiner/exe.sam/OUT_STAT/CSALD'+str(sst)+'Kfine_dx2km8192x1x128x3600_1dfile.nc'
fn2d = '/glade/scratch/linyao/SAMOUTPUT/csa2dLD'+str(sst)+'Kfine/dx2km8192x1xfiner/exe.sam/OUT_2D/CSALD'+str(sst)+'Kfine_dx2km8192x1x128x3600_2dfile.nc'
fn3d = '/glade/scratch/linyao/SAMOUTPUT/csa2dLD'+str(sst)+'Kfine/dx2km8192x1xfiner/exe.sam/OUT_3D/CSALD'+str(sst)+'Kfine_dx2km8192x1x128x3600_3dfile.nc'

ds = xr.open_dataset(fn3d)
t = ds['time']
x = ds['x'][0::4]
z = ds['z']
kx = len(x)
kz = len(z)
kt = len(t)
L = np.max(x)
dt = (t[kt-1] - t[0]) * 3600 * 24 / (kt - 1)
dx = x[1] - x[0]

T = ds['TABS'][:,:,0::4]
qn = ds['QN'][:,:,0::4] / 1000
qp = ds['QP'][:,:,0::4] / 1000
T0n = 273.16
T00n = 253.16
T0p = 283.16
T00p = 268.16

wn = (T - T00n) / (T0n - T00n)

mask = (wn > 1)
wn = xr.where(mask, 1, wn)
del mask
mask = (wn < 0)
wn = xr.where(mask, 0, wn)
del mask

wp = (T - T00p) / (T0p - T00p)
mask = (wp > 1)
wp = xr.where(mask, 1, wp)
del mask
mask = (wp < 0)
wp = xr.where(mask, 0, wp)
del mask

q_ice = (1 - wn) * qn + (1 - wp) * qp
del wn
del qn
del wp
del qp

pt = ds['PP'][:,:,0::4] + ds['p'] * 100  # total pressure
qv = ds['QV'][:,:,0::4] / 1000
Tv = (qv * eps0 + 1.0) * T
rho = pt / Tv / Rd
del Tv
del pt
rho0 = rho.mean("x")
del rho

hf = qv * lv - lf * q_ice + g * z + cp * T

del qv
del q_ice
del T

ds1 = xr.Dataset({"hf":hf})
ds1.to_netcdf('/glade/scratch/linyao/SAMOUTPUT/MSE_sst'+str(sst)+'K.nc')

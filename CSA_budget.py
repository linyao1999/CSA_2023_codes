import numpy as np 
import xarray as xr 
import os 

# set universal parameters
cp = 1004
lv = 2.5104e6
lf = 0.3336e6
Rv = 461
Rd = 287
epsilon = Rd / Rv
eps0 = 29 / 18 -1  # dry_air/H2O - 1
sec_day = 86400  # seconds in a day

# set variables
sst = int(float(os.environ["sst"]))
smooth_x = int(float(os.environ["smooth_x"]))
smooth_t = int(float(os.environ["smooth_t"]))

print(smooth_t)
print(smooth_x)

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
rho0 = rho0.rolling(time=smooth_t, min_periods=int(smooth_t/2), center=True).mean().dropna("time")  # smooth in time

Tp = T - T.mean("x")
del T
qvp = qv - qv.mean("x")
del qv
qicep = q_ice - q_ice.mean("x")
del q_ice
hfp = cp * Tp + lv * qvp - lf * qicep

del Tp
del qvp
del qicep

# smooth in time and space
#  hfp (time, z, x)
hfp = hfp.rolling(time=smooth_t, min_periods=int(smooth_t/2), center=True).mean().dropna("time")  # smooth in time
smx_bd = int((smooth_x - 1) / 2)
hfp1 = xr.concat([hfp[:,:,-smx_bd::], hfp, hfp[:,:,:smx_bd]], dim="x")
del hfp
hfp = hfp1.rolling(x=smooth_x, center=True).mean("x").dropna("x")  # smooth in x 
del hfp1

# radiative production
Q_RAD = ds['QRAD'][:,:,::4] * cp / sec_day  # change unit to J/kg/s
Q_RADp = Q_RAD - Q_RAD.mean("x")
del Q_RAD
# smooth in time and space
#  Q_RADp (time, z, x)
Q_RADp = Q_RADp.rolling(time=smooth_t, min_periods=int(smooth_t/2), center=True).mean().dropna("time")  # smooth in time
Q_RADp1 = xr.concat([Q_RADp[:,:,-smx_bd::], Q_RADp, Q_RADp[:,:,:smx_bd]], dim="x")
del Q_RADp
Q_RADp = Q_RADp1.rolling(x=smooth_x, center=True).mean("x").dropna("x")  # smooth in x 
del Q_RADp1

local_radi = Q_RADp * rho0 * hfp 
del Q_RADp
local_radi = local_radi.mean("x")

# SGS production
Qt_SGS = ds['FQT'][:,:,::4] / rho0
Qp_SGS = ds['FQP'][:,:,::4] / rho0
Qh_SGS = ds['Ft'][:,:,::4] * cp / sec_day
Q_SGS = Qt_SGS + Qp_SGS + Qh_SGS
del Qt_SGS
del Qp_SGS
del Qh_SGS
Q_SGSp = Q_SGS - Q_SGS.mean("x")
del Q_SGS
# smooth in time and space
#  Q_SGSp (time, z, x)
Q_SGSp = Q_SGSp.rolling(time=smooth_t, min_periods=int(smooth_t/2), center=True).mean().dropna("time")  # smooth in time
Q_SGSp1 = xr.concat([Q_SGSp[:,:,-smx_bd::], Q_SGSp, Q_SGSp[:,:,:smx_bd]], dim="x")
del Q_SGSp
Q_SGSp = Q_SGSp1.rolling(x=smooth_x, center=True).mean("x").dropna("x")  # smooth in x 
del Q_SGSp1

local_sgs = hfp * rho0 * Q_SGSp
del Q_SGSp
local_sgs = local_sgs.mean("x")

# FMSE variance
local_var = 0.5 * hfp * hfp * rho0
local_var = local_var.mean("x") 
del hfp
del rho0

# FMSE variance tendency
local_tend = np.zeros((kt,kz))
local_tend[1:-1,:] = 0.5 * (local_var[2:,:].values - local_var[0:-2,:].values)
local_tend[0,:] = local_var[1,:].values - local_var[0,:].values
local_tend[-1,:] = local_var[-1,:].values - local_var[-2,:].values
local_tend = local_tend / dt.values 

local_tend = xr.DataArray(
    data=local_tend,
    dims=['time','z'],
    coords=dict(
        time = local_var.time,
        z = local_var.z,
    ),
)

local_advc = local_tend - local_radi - local_sgs

ds1 = xr.Dataset({"local_tend":local_tend, "local_radi":local_radi, "local_advc":local_advc, "local_sgs":local_sgs, "local_var":local_var})
ds1.to_netcdf('FMSE_budget_1rho_sst'+str(sst)+'K_sgs_control_sx'+str(int(smooth_x*8))+'km_st'+str(int(smooth_t/24))+'d.nc')

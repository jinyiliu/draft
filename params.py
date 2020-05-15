"""Parameter file for lensing reconstruction on a idealized, full-sky simulation library.

    The CMB simulations are the FFP10 lensed CMB simulations, together with homogeneous, Gaussian noise maps.

    The CMB simulations are located on NERSC systems project directory, hence this may only be used there.

    To enable complete reconstruction, a parameter file should instantiate
        * the inverse-variance filtered simulation library 'ivfs'
        * the 3 quadratic estimator libraries, 'qlms_dd', 'qlms_ds', 'qlms_ss'.
        * the 3 quadratic estimator power spectra libraries 'qcls_dd, 'qcls_ds', 'qcls_ss'.
          (qcls_ss is required for the MCN0 calculation, qcls_ds and qcls_ss for the RDN0 calculation.
           qcls_dd for the MC-correction, covariance matrix. All three for the point source correction.)
        * the quadratic estimator response library 'qresp_dd'
        * the semi-analytical Gaussian lensing bias library 'nhl_dd'
        * the N1 lensing bias library 'n1_dd'.

    The module bandpowers.py shows how these elements are used to build the reconstructed bandpowers.

    On the first call this module will cache a couple of things will be cached in the directories defined below.

"""

import os
import healpy as hp
import numpy as np

import plancklens
from plancklens.filt import filt_simple, filt_util
from plancklens import utils
from plancklens import qest, qecl, qresp
from plancklens import nhl
from plancklens.n1 import n1
from plancklens.sims import maps, utils as maps_utils

from ali2020_sims import myCMB_sims


assert 'PLENS' in os.environ.keys(), 'Set env. variable PLENS to a writeable folder'
TEMP = os.path.join(os.environ['PLENS'], 'temp')
cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')
apomask_path = './myCMB_sims/fits/map_mask_C_2048.fits'

# Parameters
lmax_ivf = 2048
lmin_ivf = 100
lmax_qlm = 4096
nside = 2048
nlev_t = 10. # Filtering noise level in temperature (here also used for the noise simulations generation).
nlev_p = nlev_t * 2 ** 0.5 # Filtering noise level in polarization (here also used for the noise simulations generation).
nsims = 300
fwhm = 11

# Transfer function
transf = hp.gauss_beam(fwhm/ 60. / 180. * np.pi, lmax=lmax_ivf) * hp.pixwin(nside)[:lmax_ivf + 1]
cl_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))

# CMB spectra entering the QE weights (the spectra multplying the inverse-variance filtered maps in the QE legs)
cl_weight = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cl_weight['bb'] *= 0.

# Simulation library
sims = myCMB_sims()
sims = maps_utils.sim_lib_shuffle(sims, { idx : nsims if idx == -1 else idx for idx in range(-1, nsims) })


# Inverse-variance filtering library.
ftl = utils.cli(cl_len['tt'][:lmax_ivf + 1] + (nlev_t / 60. / 180. * np.pi / transf) ** 2)
fel = utils.cli(cl_len['ee'][:lmax_ivf + 1] + (nlev_p / 60. / 180. * np.pi / transf) ** 2)
fbl = utils.cli(cl_len['bb'][:lmax_ivf + 1] + (nlev_p / 60. / 180. * np.pi / transf) ** 2)
ftl[:lmin_ivf] *= 0.
fel[:lmin_ivf] *= 0.
fbl[:lmin_ivf] *= 0.

# Isotropic filtering. Independent Temp and Pol filtering
ivfs = filt_simple.library_apo_sepTP(os.path.join(TEMP, 'ivfs'), sims, apomask_path, cl_len, transf, ftl, fel, fbl, cache=True)
# ivfs = filt_simple.library_fullsky_sepTP(os.path.join(TEMP, 'ivfs'), sims, nside, transf, cl_len, ftl, fel, fbl, cache=True)

# QE libraries instances.
# For the MCN0, RDN0, MC-correction etc calculation, we need in general three of them,
# qlms_dd is the QE library which builds a lensing estimate with the same simulation on both legs
# qlms_ds is the QE library which builds a lensing estimate with a simulation on one leg and the data on the second.
# qlms_ss is the QE library which builds a lensing estimate with a simulation on one leg and another on the second.

# Shuffling dictionary.
# ss_dict remaps idx -> idx + 1 by blocks of 60 up to 300.
ss_dict = { k : v for k, v in zip( np.arange(nsims), np.concatenate([np.roll(range(i*60, (i+1)*60), -1) for i in range(0,5)]))}
ds_dict = { k : -1 for k in range(nsims) }
ivfs_d = filt_util.library_shuffle(ivfs, ds_dict) # always return data map
ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)

qlms_dd = qest.library_sepTP(os.path.join(TEMP, 'qlms_dd'), ivfs, ivfs,   cl_len['te'], nside, lmax_qlm=lmax_qlm)
qlms_ds = qest.library_sepTP(os.path.join(TEMP, 'qlms_ds'), ivfs, ivfs_d, cl_len['te'], nside, lmax_qlm=lmax_qlm)
qlms_ss = qest.library_sepTP(os.path.join(TEMP, 'qlms_ss'), ivfs, ivfs_s, cl_len['te'], nside, lmax_qlm=lmax_qlm)

# qecl libraries instances:
# This takes power spectra of the QE maps from the QE libraries, after subtracting a mean-field.

mc_sims_bias = np.arange(60) # The mean-field will be calculated from these simulations.
mc_sims_var  = np.arange(60, 300) # The covariance matrix will be calculated from these simulations

# Only qcls_dd needs a mean-field subtraction.
# mc_sims_mf is a must input of qecl.__init__
mc_sims_mf_dd = mc_sims_bias
mc_sims_mf_ds = np.array([])
mc_sims_mf_ss = np.array([])

qcls_dd = qecl.library(os.path.join(TEMP, 'qcls_dd'), qlms_dd, qlms_dd, mc_sims_mf_dd)
qcls_ds = qecl.library(os.path.join(TEMP, 'qcls_ds'), qlms_ds, qlms_ds, mc_sims_mf_ds)
qcls_ss = qecl.library(os.path.join(TEMP, 'qcls_ss'), qlms_ss, qlms_ss, mc_sims_mf_ss)

# Semi-analytical Gaussian lensing bias library:
nhl_dd = nhl.nhl_lib_simple(os.path.join(TEMP, 'nhl_dd'), ivfs, cl_weight, lmax_qlm)

# N1 lensing bias library:
libdir_n1_dd = os.path.join(TEMP, 'n1_ffp10')
n1_dd = n1.library_n1(libdir_n1_dd,cl_len['tt'],cl_len['te'],cl_len['ee'])

# QE response calculation library:
qresp_dd = qresp.resp_lib_simple(os.path.join(TEMP, 'qresp'), lmax_ivf, cl_weight, cl_len, {'t': ivfs.get_ftl(), 'e':ivfs.get_fel(), 'b':ivfs.get_fbl()}, lmax_qlm)

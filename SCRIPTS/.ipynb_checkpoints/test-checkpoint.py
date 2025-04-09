### IMPORT SCIENTIFIC LIBRARIES ###
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import curve_fit
import importlib

import matplotlib.pyplot as plt
#import seaborn as sns
plt.rcParams['font.size'], plt.rcParams['axes.labelsize'] = 14, 18
#%matplotlib widget

from tqdm import tqdm
import io, sys
from contextlib import redirect_stdout, redirect_stderr

import os

import h5py
import hdf5plugin

import sys
sys.path.append('./XPCS_library/')

import ID10_tools as ID10
from ID10_tools import Nx, Ny, Npx
importlib.reload(ID10)
ID10.set_version('v2')

import XPCS_tools as XPCS
from XPCS_tools import E2lambda, lambda2E, theta2Q, Q2theta, decorelation_f
importlib.reload(XPCS)
XPCS.set_beamline('ID10')
XPCS.set_expvar(Nx//2, Ny//2, 7)

import COSMICRAY_tools as COSMIC
importlib.reload(COSMIC)
COSMIC.set_beamline('ID10')

# RAW FOLDER PATH
raw_folder = '../RAW_DATA/'
masks_folder = '../masks/'


e4m_htmask = np.load(masks_folder+'e4m_htmask_copper_foil_30um_1_1'+'.npy')
e4m_mask = np.load(masks_folder+'e4m_mask'+'.npy')
bs_mask = np.load(masks_folder+'bs_mask_copper_foil_30um'+'.npy')

#######################################
sample_name = 'vycor'
Ndataset = 1
Nscan = 7
ID10.Nfmax_dense_file = 10000
#######################################

scan = ID10.load_scan(raw_folder, sample_name, Ndataset, Nscan)
Ei = scan['monoe']
itime = scan['fast_timer_period'][0]
theta = scan['delcoup']
Q = round(XPCS.theta2Q(Ei,  theta),2)

print('#############################')
print('command =', scan['command'])
print('Ei =', Ei)
print('itime =', itime)
print('theta =', theta)
print('Q =', Q)
print('#############################\n')

Nfi, Nff = 0, 100
e4m_data = ID10.load_dense_e4m_new(raw_folder, sample_name, Ndataset, Nscan, Nfi, Nff,  n_jobs=1, tosparse=False)

geom = [{'geom':'Rectangle', 'x0':1100, 'y0':1100, 'xl':550, 'yl':1950, 'inside':False},
        {'geom':'Rectangle', 'x0':0, 'y0':0, 'xl':80, 'yl':80, 'inside':False},
        {'geom':'Rectangle', 'x0':0, 'y0':1600, 'xl':300, 'yl':2000, 'inside':False}     
        ]
XPCS.set_expvar(1350, 1400, 7)

XPCS.gen_plots4mask(e4m_data, itime, Ith_high=5000, Nff=4, mask_geom=geom,)

bs_mask = XPCS.gen_mask(e4m_data, itime, e4m_mask=None, mask_geom=geom, hist_plots=False)

Qmask = XPCS.gen_Qmask(Ei, theta, .06, .01, Qmap_plot=True)

###################
Nfi = None
Nff = None
Lbin = 1
mask=e4m_mask*e4m_htmask*bs_mask*Qmask
###################

G2t = XPCS.get_G2t(e4m_data, mask=mask, Nfi=Nfi, Nff=Nff, Lbin=None)
t, g2 = XPCS.get_g2(itime*Lbin, G2t, cython=False)
t_mt, g2_mt = XPCS.get_g2_mt(itime*Lbin, g2)
# load needed modules
import numpy as np

import os.path
import shutil
import re
import cmath

from triqs_dft_tools.sumk_dft import *
from triqs_dft_tools.sumk_dft_tools import *
from triqs_dft_tools.converters.vasp_converter import *
from triqs_dft_tools.converters.plovasp.vaspio import VaspData
from triqs_dft_tools.converters.plovasp.plotools import generate_plo, output_as_text
import triqs_dft_tools.converters.plovasp.converter as plo_converter

from pytriqs.gf import *
import pytriqs.utility.mpi as mpi 

### parameter section
broadening = 0.1



### start ##########

with HDFArchive('rot_mat.h5','r') as ar:
    block_s = ar['DMFT_input']['block_structure']
    rot_mat_den = ar['DMFT_input']['rot_mat']

sum_k = SumkDFT(hdf_file = 'vasp.h5' , use_dft_blocks = False)
sum_k.set_mu(0.008405)

corr_to_inequiv = sum_k.corr_to_inequiv

sum_k.block_structure = block_s
sum_k.rot_mat = rot_mat_den

sum_k.corr_to_inequiv = corr_to_inequiv


Sigma = sum_k.block_structure.create_gf(gf_function=GfReFreq, window = (-10,10), n_points = 6001 )
sum_k.put_Sigma([Sigma])

mpi.report("setup okay extracting Gloc")
G_loc_all = sum_k.extract_G_loc(iw_or_w='w',broadening = broadening)
den_mat_dft = G_loc_all[0].density()


delta = [sum_k.block_structure.create_gf(gf_function=GfReFreq, window = (-10,10), n_points = 6001 )]

mpi.report("obtaining local Hamiltonian")
atomic_levels = sum_k.eff_atomic_levels()

mpi.report("extracting delta now")
for i, shell in enumerate(G_loc_all):
    for name, g0_w in shell:
        print(i, name)
        n = name.split('_')
        atomic_level = atomic_levels[i][n[0]]
        delta[i][name] << Omega + 1j*broadening - inverse(g0_w)
        delta[i][name] = delta[i][name] - atomic_level


mpi.report("storing to h5")

if mpi.is_master_node():
    # store
    h5 = HDFArchive('FeAlN_dproj_inp_rotden_999_6k_eta_0.1.h5','a')

    if not 'DMFT_input' in h5:
        h5.create_group('DMFT_input')

    h5['DMFT_input']['block_structure'] = block_s
    h5['DMFT_input']['rot_mat'] =  rot_mat_den
    h5['DMFT_input']['G_loc_all'] = G_loc_all
    h5['DMFT_input']['delta'] = delta
    h5['DMFT_input']['atomic_levels'] = atomic_levels
    h5['DMFT_input']['den_mat_dft'] = den_mat_dft

    del h5
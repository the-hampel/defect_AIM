#!/usr/bin/python

# system
import os
import numpy as np

# triqs
from triqs.gf import *
from h5 import HDFArchive

# fork tps
import forktps as ftps
from forktps.DiscreteBath import *
from forktps.Helpers import *


#### parameter section
h5_in = 'FeAlN_delta_inp_rot_999_10k.h5'
gfstruct = [ ['up',[0,1,2,3,4]], ['dn',[0,1,2,3,4]]]





#### parameter section end



with HDFArchive(h5_in,'r') as ar:
    delta_in = ar['DMFT_input']['delta'][0]

# create input delta for solver with correct up / down block names
# (need to be named up / down)
dup = delta_in['up_0'].copy()
ddn = delta_in['down_0'].copy()
Delta = BlockGf(name_list = ('up','dn'), block_list = (dup,ddn), make_copies = False)

for block, hyb_blocks in Delta:
        shp = hyb_blocks.target_shape
        for i_orb in range(0,shp[0]):
            for j_orb in range(0,shp[1]):
                if i_orb != j_orb:
                    hyb_blocks[i_orb,j_orb] << 0.0+0.0j

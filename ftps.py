#!/usr/bin/python

# system
import os
import numpy as np

# triqs
from triqs.gf import *
from h5 import HDFArchive
import triqs.utility.mpi as mpi
from triqs_dft_tools.sumk_dft import SumkDFT

# fork tps
import forktps as ftps
from forktps.DiscreteBath import *
from forktps.Helpers import *


# nice format for numpy print
np.set_printoptions(precision=4,suppress=False)
#### parameter section
h5_in = 'FeAlN_delta_inp_rot_999_10k.h5'
h5_out = 'FeAlN_nb9.h5'
gfstruct = [ ['up',[0,1,2,3,4]], ['dn',[0,1,2,3,4]]]

hyb_diag = True
hint_diag = True
hloc_diag = True
norb = 5

# interaction parameters
U = 4.143
U_p = 2.714
J = 0.714
dc = True
dc_N = 6.5

# FTPS solver parameters

Nb = 9  # number of bath sites
bath_gap = [-1.8,2.4]
eta = 0.05 # broadening
nw = 5001
wmin = -10.0
wmax = 10.0

params_DMRG = ftps.solver.DMRGParams(
        maxmI=200,  # max bond dimension imp-imp  (upper bound)
        maxmIB=200,  # imp-bath
        maxmB=300,   # bath-bath
        twI=1E-12, # truncated weights for imp bond dimension
        twIB=1E-12,# -"- imp-bond
        twB=1E-12,  # -"- bath-bath links
        nmax=2,   # krylov space size, not too large ~5, to not end in local minima
        normErr=1E-10, # norm err of new krylov vector to stop
        conv=1E-12,  # for eigenvalue of "local" GS
        sweeps=25,   # 20-30 , 1 sweep, once through the lattice
        napph = 2) # help to find GS if DMRG is stuck

params_tevo = ftps.solver.TevoParams(
        dt=0.1,  # size of time step
        method='TDVP',  # method for time evolution, for TEBD smaller time steps
        time_steps=0, # 150 now until t=30, half ot total time steps that are performed
        maxmI=200, # following same as above or a bit stricter / larger
        maxmIB=200,
        maxmB=300,
        twI=1E-10,
        twIB=1E-10,
        twB=1E-10,
        nmax=80, # krylov parameter for Tevo, those should be way stricter than in DMRG
        normErr=1E-10, # should be not too smal
        conv=1E-12 )

things_to_save = ['gfstruct', 'norb', 'Nb', 'bath_gap', 'eta','Hloc_0_DC', 'U', 'J', 'U_p', 'dc_N', 'dc_pot', 'params_tevo', 'params_DMRG', 'Delta', 'DeltaDiscrete' ]

#### parameter section end

#### helper functions
def calc_dc_kanamori(U, J, n, M):
    r"""
    Held DC formula for cubic symmetry U_prime=U-2J
    """
    dc_pot = np.identity(M)

    val = (U + (M - 1) * (U - 2.0 * J) + (M - 1)
          * (U - 3.0 * J)) / (2 * M - 1) * (n - 0.5)

    dc_pot *= val

    return dc_pot


# read input delta from given h5 archive
if mpi.is_master_node():
    with HDFArchive(h5_in,'r') as ar:
        delta_in = ar['DMFT_input']['delta'][0]
        atomic_levels = ar['DMFT_input']['atomic_levels'][0]
        den_mat_dft = ar['DMFT_input']['den_mat_dft']

# create input delta for solver with correct up / down block names
# (need to be named up / down)
dup = delta_in['up_0'].copy()
ddn = delta_in['down_0'].copy()
Delta = BlockGf(name_list = ('up','dn'), block_list = (dup,ddn), make_copies = False)

# set off diag to zero
if hyb_diag == True:
    for block, hyb_blocks in Delta:
        shp = hyb_blocks.target_shape
        for i_orb in range(0,shp[0]):
            for j_orb in range(0,shp[1]):
                if i_orb != j_orb:
                    hyb_blocks[i_orb,j_orb] << 0.0+0.0j

mpi.report('\ndiscretizing bath...')

### create bath object
bath = DiscretizeBath(Delta=Delta, NBath=Nb, gap = bath_gap )

# discretize bath
DeltaDiscrete = bath.reconstructDelta( w = ddn.mesh, eta = eta )

### setup Hloc_0
#give the local Hamiltonian the right block structure
Hloc_0 = ftps.solver_core.Hloc( MakeGFstruct(Delta)  )

if dc:
    mpi.report('impurity occupation for DC determination: '+"{:1.4f}".format(dc_N))
    mpi.report('calculating DC:')
    dc_pot = calc_dc_kanamori(U, J, dc_N, norb)
    mpi.report(dc_pot)
else:
    dc_pot = 0.0

for sp, mat in atomic_levels.items():
    if 'up' in sp:
        hloc_dft = mat.real
        # make diagonal if wanted
        if hloc_diag:
            hloc_dft = np.diag(np.diag(hloc_dft))
        mpi.report('\nDFT local Hamiltonian:')
        mpi.report(hloc_dft)


# subract DC
Hloc_0_DC = hloc_dft - dc_pot

mpi.report('\nHloc_0 entering solver:')
mpi.report(hloc_dft-dc_pot)

# fill Hloc_0 FTPS object
Hloc_0.Fill(Hloc_0_DC)


### create Hint
mpi.report('\ncreate Kanamori interaction matrix with U='+str(U)+' J='+str(J)+' and Up='+str(U_p)+' eV')
Hint = ftps.solver_core.HInt(u=U,j=J, up=U_p, dd = hint_diag)


# create solver
S = ftps.Solver(gf_struct = gfstruct , nw = nw, wmin=wmin, wmax=wmax)
# Other TRIQS solver usually have the Weiss field as member. It carries the exact same information as the
# bath and e0 for the ForkTPS solver
S.b = bath
S.e0 = Hloc_0

mpi.report('\nsetup okay. Storing input to h5: '+h5_out)

# before the run store all input parameters
if mpi.is_master_node():
    with HDFArchive(h5_out,'a') as ar:
    	for it in things_to_save: ar[it] = locals()[it]


S.solve(  h_int = Hint,
#          calc_me = [['up',0]], # only calculate a specific orbital in the tevo
          params_GS = params_DMRG,
          params_partSector = params_DMRG,
          tevo=params_tevo,
#           NPart = [[3,3,3,3]], # list of list containing N sector per spin orbital for whole system (bath + impurity)
          do_self_energy = False)


if mpi.is_master_node():
    with HDFArchive(h5_out,'a') as ar:
        ar['solver'] = S


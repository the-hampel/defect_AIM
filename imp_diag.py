import os
from itertools import product
import numpy as np
import sympy as sp

from h5 import HDFArchive
from triqs_dft_tools.sumk_dft import SumkDFT
import triqs.utility.mpi as mpi

from triqs.operators import *
from triqs.operators.util.observables import *
from triqs.operators.util.U_matrix import U_matrix, transform_U_matrix
from triqs.operators.util.hamiltonians import h_int_slater, make_operator_real
from triqs.gf import *
from triqs.atom_diag import AtomDiag, quantum_number_eigenvalues, atomic_g_w, atomic_density_matrix, trace_rho_op

################################################
### parameter section
h5_input = '/mnt/home/ahampel/work/defect/FeAlN/proj_run_tet_555/vasp.h5'
block_structure_h5 = '/mnt/home/ahampel/work/defect/FeAlN/run_test/rot_den/vasp.h5'
mu_init = 0.004132
U = 4.0
J = 0.7

# adjust mu to obtain 6 electron
mu =14
# U=3 : mu=14
# U=4 : mu=18
# U=5 : mu=24

spin_names = ['up','down']
beta = 200

store_h5 = True
store_h5_file = 'imp_diag.h5'

################################################

# For sorting
def take_second(elem):
    return elem[1]

# Print out information about the states
def write_evs(spin_names, orb_names, ad, target_mu, prt_state=True):
    '''
    Sort eigenstates and write them in fock basis

    Inputs:
    orb_names: List of orbitals
    spin_names: List of spins
    ad: Solution to the atomic problem
    target_mu: Only keep states with a given occupation

    Outputs:
    eigensys: List containing states, energies, and other info

    '''
    # Get spin eigenvalues, may be useful
    S2 = S2_op(spin_names, orb_names, off_diag=True)
    S2 = make_operator_real(S2)
    Sz=S_op('z', spin_names, orb_names, off_diag=True)
    Sz = make_operator_real(Sz)


    S2_states = quantum_number_eigenvalues(S2, ad)
    Sz_states = quantum_number_eigenvalues(Sz, ad)

    n_orb=len(orb_names)

    n_eigenvec=0.0
    eigensys=[]
    # Loop through subspaces
    for sub in range(0,ad.n_subspaces):
        skip_sub=False
        # get fock state in nice format
        subspace_fock_state=[]
        for fs in ad.fock_states[sub]:
            state =int(bin(int(fs))[2:])

            state_leng=n_orb*2
            fmt='{0:0'+str(state_leng)+'d}'
            state_bin="|"+fmt.format(state)+">"
            state_bin_sym=sp.symbols(state_bin)
            subspace_fock_state.append(state_bin_sym)

        if skip_sub:
            continue

        # convert to eigenstate
        kp = sp.kronecker_product
        u_mat=sp.Matrix(np.round(ad.unitary_matrices[sub],10).conj().T)
        st_mat=sp.Matrix(subspace_fock_state)
        eig_state=np.matrix(u_mat*st_mat)
        # format eigenstate:
        sub_state=[]
        for row in eig_state:
            sub_state.append(str(row).replace('[','').replace(']',''))

        # store state and energy
        for ind in range(0,ad.get_subspace_dim(sub)):

            eng=ad.energies[sub][ind]
            spin=round(float(S2_states[sub][ind]),3)
            ms=round(float(Sz_states[sub][ind]),3)
            eigensys.append([sub_state[ind],eng,spin,n_eigenvec,ms.real])

            # Keep track of the absolute number of eigenstate
            n_eigenvec += 1


    # Sort by energy
    eigensys.sort(key=take_second)

    # print info for given number of states
    f_state= open("eigensys.dat","w+")

    for ii in range(len(eigensys)):
        # Write out info
        f_state.write('%10s %10s %10s %10s %10s %10s %10s %10s\n' % \
                      ("Energy:", np.round(float(eigensys[ii][1]),6), \
                       "HS eig num:",eigensys[ii][3],\
                       "s(s+1):", eigensys[ii][2], \
                       "ms:",np.round(float(eigensys[ii][4]),3)))
                       #"ml:",np.round(float(eigensys[ii][5]),3),\
                       #"l(l+1):",np.round(float(eigensys[ii][6]))))
        if prt_state:
            f_state.write('%10s %s\n' % ("State:",eigensys[ii][0]))
            f_state.write("\n ")

    f_state.close()
    return eigensys


# load block structure and rotation matrices
with HDFArchive(block_structure_h5,'r') as ar:
    block_s = ar['DMFT_input']['block_structure']
    rot_mat_den = ar['DMFT_input']['rot_mat']

# load projector input from vasp_converter
sum_k = SumkDFT(hdf_file = h5_input , use_dft_blocks = False)
sum_k.set_mu(mu_init)

corr_to_inequiv = sum_k.corr_to_inequiv
# set block structure
sum_k.block_structure = block_s

# workaround since the old dfttools did not save the corr_to_inequiv variable correctly
sum_k.corr_to_inequiv = corr_to_inequiv
rot_mat = rot_mat_den

# set an empty sigma into sumk
broadening = 0.01
Sigma = sum_k.block_structure.create_gf(gf_function=GfReFreq, window = (-10,10), n_points = 5001 )
sum_k.put_Sigma([Sigma])

# orb name def
n_orb = sum_k.corr_shells[0]['dim']
orb_names = list(range(n_orb))

###
mpi.report('######################\n')
mpi.report('Sum_k setup okay - move it')


# check for previous calculation
things_to_load = ['Gloc', 'Hloc_0', 'Hint', 'atom_diag','eigensys']
if os.path.exists(store_h5_file):
    with HDFArchive(store_h5_file,'r') as ar:
            for elem in things_to_load:
                if elem in ar:
                    mpi.report('loading '+elem+' from archive '+ store_h5_file)
                    vars()[elem] = ar[elem]

# extract
# if not 'Gloc' in locals():
    # mpi.report('extracting Gloc')
    # Gloc = sum_k.extract_G_loc(iw_or_w='w',broadening = broadening)[0]

# np.set_printoptions(precision=4, suppress= True)
# density_shell = np.real(Gloc.total_density())
# mpi.report('Total charge of impurity problem = {:.4f}'.format(density_shell))
# density_mat = Gloc.density()
# mpi.report('Density matrix:')
# for key, value in density_mat.items():
    # mpi.report(key)
    # mpi.report(np.real(value))


# Setup the particle number operator
N = Operator() # Number of particles
for s in spin_names:
    for o in orb_names:
        N += n(s,o)

# get local Hamiltonian
if not 'Hloc_0' in locals():
    mpi.report('calculating local Hamiltonian')
    atomic_levels = sum_k.eff_atomic_levels()[0]

    # Make noniteracting operator
    Hloc_0=Operator()
    for spin in spin_names:
        for o1 in orb_names:
            for o2 in orb_names:
                Hloc_0 += atomic_levels[spin][o1,o2] * (c_dag(spin,o1) * c(spin,o2)
                                                       +c_dag(spin,o2) * c(spin,o1)
                                                       )
    Hloc_0 += (-mu) *N


# store data obtained
if store_h5 and mpi.is_master_node():
    with HDFArchive(store_h5_file,'a') as ar:
        ar['block_structure'] = sum_k.block_structure
        ar['mu'] = sum_k.chemical_potential
        # ar['Gloc'] = Gloc
        ar['Hloc_0'] = Hloc_0

# construct interaction matrix
if not 'Hint' in locals():
    mpi.report('setting up slater Hamiltonian')
    Umat_full = U_matrix(l=2, U_int=U, J_hund=J, basis='cubic')

    # rotate to den mat diag basis
    Umat_full_rotated = transform_U_matrix(Umat_full, sum_k.rot_mat[0].T)

    # create interaction Hamiltonian
    Hint = h_int_slater(spin_names, orb_names, off_diag=True, U_matrix=Umat_full_rotated)

    # store data obtained
    if store_h5 and mpi.is_master_node():
        with HDFArchive(store_h5_file,'a') as ar:
            ar['Hint'] = Hint

# setting up full Hamiltonian
Hloc_full = Hloc_0 + Hint

# fundemental operators
fops = [(sn,on) for sn, on in product(spin_names,orb_names)]

# AtomDiag
if not 'atom_diag' in locals():
    atom_diag = AtomDiag(Hloc_full, fops)

    # store data obtained
    if store_h5 and mpi.is_master_node():
        with HDFArchive(store_h5_file,'a') as ar:
            ar['atom_diag'] = atom_diag

beta = 1e5
dm = atomic_density_matrix(atom_diag, beta)
filling = trace_rho_op(dm, N, atom_diag)
mpi.report('atom diag occupation: '+'{:.4f}'.format(filling.real))

# get eigenenergies
evs = []
for sub in range(0,atom_diag.n_subspaces):
    for fs in atom_diag.fock_states[sub]:
        for ind in range(0,atom_diag.get_subspace_dim(sub)):
            ev=atom_diag.energies[sub][ind]
            evs.append(ev)

if mpi.is_master_node() and not 'eigensys' in locals():
    eigensys = write_evs(spin_names, orb_names, atom_diag, sum_k.density_required, prt_state=True)

# store data obtained
if store_h5 and mpi.is_master_node():
    with HDFArchive(store_h5_file,'a') as ar:
        ar['evs'] = np.array(evs)
        ar['gs_energy'] = atom_diag.gs_energy
        ar['eigensys'] = eigensys

# Atomic Green's functions
gf_struct = [['down',orb_names],['up',orb_names]]
G_w = atomic_g_w(atom_diag, beta, gf_struct, (-10, 10), 5001, 0.01)

# store G_w
if store_h5 and mpi.is_master_node():
    with HDFArchive(store_h5_file,'a') as ar:
        ar['G_w'] = G_w.copy()




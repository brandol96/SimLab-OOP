import os
from SimLab_OOP.utils import path_dftb


def run(method, path, step, fermiFilling, mol, mol_name):
    if method == 'DFTB':
        from ase.calculators.dftb import Dftb
        sampling = path_dftb(path, step, mol, True, False)
        print(sampling)
        bands = Dftb(atoms=mol,
                     label='dftb_band_out',
                     Hamiltonian_KPointsAndWeights=f'KLines {{ \n {sampling} }}',
                     Hamiltonian_SCC='Yes',
                     Hamiltonian_SCCTolerance='1e-7',
                     Hamiltonian_ReadInitialCharges='Yes',
                     Hamiltonian_MaxSCCIterations='1',
                     Hamiltonian_Filling=f"Fermi{{Temperature [K] = {fermiFilling} }}")

        # run calculation through DFTB+ implemented routines
        bands.calculate(mol)

        # dp_band to transform band.out into plottable data
        os.system(f'dp_bands band.out {mol_name}.band')
        os.system(f'dp_dos band.out {mol_name}.dos.dat')
import os
import SimLab.calculators
from mpi4py import MPI
from ase.io import read
from ase.io import write
from SimLab.utils import parprint
from SimLab.utils import cleanup

def run(method, mol, calc):
    # run optimization through DFTB+ implemented routines
    if method == 'DFTB':
        mol.calc = calc
        calc.calculate(mol)

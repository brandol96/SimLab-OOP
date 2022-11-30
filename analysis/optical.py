import os
import shutil
from mpi4py import MPI
from ase.io import read
from SimLab.utils import parprint
from SimLab.utils import cleanup


def run_kick(mol, maxSCC, maxSCCSteps, fermiFilling, totalTime, timeStep, fieldStrength, direction):
    from ase.calculators.dftb import Dftb

    # writing DFTB electron dynamics manually
    totalSteps = int(totalTime / timeStep)

    electron_dynamics = '{'
    electron_dynamics += f'\nSteps = {totalSteps}'
    electron_dynamics += f'\nTimeStep [fs] = {timeStep}'
    electron_dynamics += f'\nPerturbation = Kick {{ \nPolarizationDirection = {direction}\n }} \n'
    electron_dynamics += f'FieldStrength [V/A] = {fieldStrength}\n}}'

    optical = Dftb(atoms=mol,
                   label=f'optical_run_{direction}',
                   Hamiltonian_SCC='Yes',
                   Hamiltonian_SCCTolerance=maxSCC,
                   Hamiltonian_ReadInitialCharges='Yes',
                   Hamiltonian_MaxSCCIterations=maxSCCSteps,
                   Hamiltonian_Filling=f"Fermi{{Temperature [K] = {fermiFilling} }}",
                   ElectronDynamics=electron_dynamics)

    # run calculation through DFTB+ implemented routines
    optical.calculate(mol)


def run_kick_dftb(direction, calc_base, details):
    # setup
    input_list = os.listdir()
    curr_ase_dftb_command = os.environ["ASE_DFTB_COMMAND"]
    os.environ["ASE_DFTB_COMMAND"] = "dftb+ | tee PREFIX.out"
    for inputFile in input_list:
        if '.traj' in inputFile:
            mol_name = os.path.splitext(os.path.basename(inputFile))[0]
            out_path = f'Optimize_DFTB-{calc_base}_{mol_name}' + os.sep

            try:
                mol = read(f'{out_path}DFTB-{calc_base}_{mol_name}_end.traj')
            except NameError:
                parprint(f'Please, run optimization for {mol_name}')
            # set sampling

            pbc = mol.get_pbc()
            if True in pbc:
                if details['verbose']:
                    print('\n\nObservation: Some direction has pbc, as far as I know DFTB cannot properly calculate '
                          'optical stuff for periodic systems \n\n')
                return 'molecule is NOT a cluster'
            else:
                if details['verbose']:
                    print('\n\nObservation: No direction has pbc, following with optical calculations\n\n')
                details['cluster'] = True

            from ase.calculators.dftb import Dftb
            from os.path import abspath
            shutil.copyfile(f'{out_path}charges.bin', f'{os.getcwd()}{os.sep}charges.bin')

            # writing DFTB electron dynamics manually
            tot_time = details['totalTime']
            time_step = details['timeStep']
            tot_steps = int(tot_time / time_step)
            field_strength = details['fieldStrength']

            electron_dynamics = '{'
            electron_dynamics += f'\nSteps = {tot_steps}'
            electron_dynamics += f'\nTimeStep [fs] = {time_step}'
            electron_dynamics += f'\nPerturbation = Kick {{ \nPolarizationDirection = {direction}\n }} \n'
            electron_dynamics += f'FieldStrength [V/A] = {field_strength}\n}}'

            optical = Dftb(atoms=mol,
                           label=f'optical_run_{direction}',
                           Hamiltonian_SCC='Yes',
                           Hamiltonian_SCCTolerance=details['maxSCC'],
                           Hamiltonian_ReadInitialCharges='Yes',
                           Hamiltonian_MaxSCCIterations='1000',
                           Hamiltonian_Filling=f"Fermi{{Temperature [K] = {details['fermiFilling']} }}",
                           ElectronDynamics=electron_dynamics)
            parprint(f'\n\nCalculating Optical Absorption "Kick" type field'
                     f'\nMethod: {calc_base}\nInput molecule: {inputFile}'
                     f'\nDirection: {direction}')

            # run calculation through DFTB+ implemented routines
            optical.calculate(mol)

            # cleanup
            MPI.COMM_WORLD.Barrier()
            cleanup(out_path)
    os.environ["ASE_DFTB_COMMAND"] = curr_ase_dftb_command


def run_laser_dftb(direction, calc_base, details):
    # setup
    input_list = os.listdir()
    curr_ase_dftb_command = os.environ["ASE_DFTB_COMMAND"]
    os.environ["ASE_DFTB_COMMAND"] = "dftb+ | tee PREFIX.out"
    for inputFile in input_list:
        if '.traj' in inputFile:
            mol_name = os.path.splitext(os.path.basename(inputFile))[0]
            out_path = f'Optimize_DFTB-{calc_base}_{mol_name}' + os.sep

            try:
                mol = read(f'{out_path}DFTB-{calc_base}_{mol_name}_end.traj')
            except NameError:
                parprint(f'Please, run optimization for {mol_name}')
            # set sampling

            pbc = mol.get_pbc()
            if True in pbc:
                if details['verbose']:
                    print('\n\nObservation: Some direction has pbc, as far as I know DFTB cannot properly calculate '
                          'optical stuff for periodic systems \n\n')
                return 'molecule is NOT a cluster'
            else:
                if details['verbose']:
                    print('\n\nObservation: No direction has pbc, following with optical calculations\n\n')
                details['cluster'] = True

            from ase.calculators.dftb import Dftb
            from os.path import abspath
            shutil.copyfile(f'{out_path}charges.bin', f'{os.getcwd()}{os.sep}charges.bin')

            # writing DFTB electron dynamics manually
            tot_time = details['totalTime']
            time_step = details['timeStep']
            tot_steps = int(tot_time / time_step)
            field_strength = details['fieldStrength']
            laser_energy = details['laserEnergy']

            electron_dynamics = '{'
            electron_dynamics += f'\nSteps = {tot_steps}'
            electron_dynamics += f'\nTimeStep [fs] = {time_step}'
            electron_dynamics += f'\nPerturbation = Laser {{ \nPolarizationDirection = {direction[0]} {direction[1]} {direction[2]}'
            electron_dynamics += f'\nLaserEnergy [eV] ={laser_energy} }}'
            electron_dynamics += f'\nFieldStrength [V/A] = {field_strength}\n}}'

            optical = Dftb(atoms=mol,
                           label=f'optical_run_laser',
                           Hamiltonian_SCC='Yes',
                           Hamiltonian_SCCTolerance=details['maxSCC'],
                           Hamiltonian_ReadInitialCharges='Yes',
                           Hamiltonian_MaxSCCIterations='1000',
                           Hamiltonian_Filling=f"Fermi{{Temperature [K] = {details['fermiFilling']} }}",
                           ElectronDynamics=electron_dynamics)
            parprint(f'\n\nCalculating Optical Absorption "Laser" type field'
                     f'\nMethod: {calc_base}\nInput molecule: {inputFile}'
                     f'\nDirection: {direction}')

            # run calculation through DFTB+ implemented routines
            optical.calculate(mol)

            # cleanup
            MPI.COMM_WORLD.Barrier()
            cleanup(out_path)
        os.environ["ASE_DFTB_COMMAND"] = curr_ase_dftb_command

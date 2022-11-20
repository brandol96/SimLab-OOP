# file manipulation libraries
import os
from ase.io import read
from ase.io import Trajectory
from mpi4py import MPI
from ase.optimize import QuasiNewton
from SimLab.utils import cleanup
import SimLab.calculators
from SimLab.utils import parprint
from SimLab.utils import read_energy_dftb


def run(method, maxCauchyStrain, totalSteps, calc, original_mol, out_path, mol_name):
    step = maxCauchyStrain / totalSteps
    directions = ['XX', 'YY', 'XY']

    if method == 'DFTB':
        for direction in directions:
            # important quantities to be reset at each direction
            traj = Trajectory(f'{out_path}{method}_{mol_name}_{direction}.traj', 'w')
            mol = original_mol.copy()
            cell = mol.get_cell()
            lx = cell[0][0]
            ly = cell[1][1]
            en = 0.0
            eps = -maxCauchyStrain
            i = 1
            mol.set_pbc(True)

            # Strain definitions
            energy = ['Energy']
            strain_x = ['Strain_x']  # Lx-lx
            strain_y = ['Strain_y']  # Ly-ly
            strain_per_x = ['strain_per_x']  # Lx/lx
            strain_per_y = ['strain_per_y']  # Ly/ly
            cauchy_s = ['CauchyStrain']  # StrainX/lx or StrainY/ly.

            print(f'\n #### STRAIN IN {direction} DIRECTION ####')

            while round(eps, 6) <= maxCauchyStrain:
                cauchy_s.append(round(eps, 6))

                if 'X' in direction:
                    cell[0][0] = (1 + eps) * lx
                strain_x.append(round(cell[0][0] - lx, 6))
                strain_per_x.append(round(cell[0][0] / lx, 6))

                if 'Y' in direction:
                    cell[1][1] = (1 + eps) * ly
                strain_y.append(round(cell[1][1] - ly, 6))
                strain_per_y.append(round(cell[1][1] / ly, 6))

                # apply current step setting and calculator
                mol.set_cell(cell)
                mol.center()
                mol.set_calculator(calc)

                # Optimize geometry for current cell with GPAW optimization
                # opt = QuasiNewton(mol, trajectory=f'{out_path}temp.traj',logfile='BFGS.log', )  # enable to write a file with GPAW entire output
                # opt.run(fmax=fmax, steps=1000)  # run optimization until fmax is reached

                # instead of usng ASE routines I'll to leaving it to DFTB
                calc.calculate(mol)
                MPI.COMM_WORLD.Barrier()

                # read optimization output after all nodes are done
                # mol = read(f'{out_path}temp.traj')
                # en = mol.get_potential_energy()

                mol = read('geo_end.gen')
                en = read_energy_dftb()

                # remove temp.traj file after all nodes got the mol
                MPI.COMM_WORLD.Barrier()

                # output
                traj.write(mol)
                energy.append(en)

                MPI.COMM_WORLD.Barrier()

                parprint('Original length x  |' + str(round(lx, 6)) + ' A')
                parprint('Original Length y  |' + str(round(ly, 6)) + ' A')
                parprint('Original Cell Area |' + str(round(lx * ly, 6)) + ' A²')
                parprint('New Length x       |' + str(round(cell[0][0], 6)) + ' A')
                parprint('New Length y       |' + str(round(cell[1][1], 6)) + ' A')
                parprint('Strain x           |' + str(round(strain_x[i], 6)) + ' A')
                parprint('Strain y           |' + str(round(strain_y[i], 6)) + ' A')
                parprint('Stretch Ratio x    |' + str(round(strain_per_x[i], 6)) + ' A')
                parprint('Stretch Ratio y    |' + str(round(strain_per_y[i], 6)) + ' A')
                parprint('#######################')
                parprint('Cauchy Strain      |' + str(round(eps, 6)))
                parprint('Total Energy       |' + str(round(en, 6)) + ' eV')
                parprint('Energy per Area    |' + str(round(en / (lx * ly), 6)) + ' eV/A²')
                parprint('\n')
                i += 1
                eps += step

            # write output file
            import csv
            import numpy
            data = numpy.array(
                [['Cell_vector_X'] + [lx] * (i - 1), ['Cell_Vector_Y'] + [ly] * (i - 1), strain_x, strain_y,
                 strain_per_x, strain_per_y, cauchy_s, energy])
            data = data.transpose()
            with open(f'{method}_{mol_name}_{direction}.csv', 'w+') as out:
                write = csv.writer(out)
                write.writerows(data)

            # cleanup
            MPI.COMM_WORLD.Barrier()
            cleanup(out_path)


def run_dftb(method, calc_base, details):
    # setup of constant variables for all files and directions
    maxCauchyStrain = details['maxCauchyStrain']
    totalSteps = details['totalCauchySteps']
    details['lattice'] = 'No'
    inputList = os.listdir()
    step = maxCauchyStrain / totalSteps
    directions = ['XX', 'YY', 'XY']
    fmax = details['maxforce']

    # setup environment variables to ensure clean screen print
    curr_ase_dftb_command = os.environ["ASE_DFTB_COMMAND"]
    os.environ["ASE_DFTB_COMMAND"] = "dftb+ > PREFIX.out"

    for inputFile in inputList:
        if '.traj' in inputFile:
            # setup of variables depending of which file is currently being executed
            mol_name = os.path.splitext(os.path.basename(inputFile))[0]
            out_path = f'Optimize_DFTB-{calc_base}_{mol_name}' + os.sep

            try:
                original_mol = read(f'{out_path}DFTB-{calc_base}_{mol_name}_end.traj')
            except NameError:
                parprint(f'Please, run optimization for {mol_name}')

            # Not allowing GPAW because I must redo this calculator call
            # if method=='GPAW': calc = SimLab.calculators.setup_GPAW(calc_base,details) #call chosen calculator

            if method == 'DFTB': calc = SimLab.calculators.setup_dftb(calc_base, details)  # call chosen calculator

            for direction in directions:
                # important quantities to be reset at each direction
                traj = Trajectory(out_path + f'{method}-{calc_base}_{mol_name}_{direction}.traj', 'w')
                mol = original_mol.copy()
                cell = mol.get_cell()
                lx = cell[0][0]
                ly = cell[1][1]
                en = 0.0
                eps = -maxCauchyStrain
                i = 1
                mol.set_pbc(True)

                # Strain definitions
                energy = ['Energy']
                strain_x = ['Strain_x']  # Lx-lx
                strain_y = ['Strain_y']  # Ly-ly
                strain_per_x = ['strain_per_x']  # Lx/lx
                strain_per_y = ['strain_per_y']  # Ly/ly
                cauchy_s = ['CauchyStrain']  # StrainX/lx or StrainY/ly.

                parprint('\n\nRunning Elastic Analysis \nMethod: ' + method + '\nInput molecule: ' + inputFile + '\n')
                parprint('\n #### STRAIN IN ' + direction + ' DIRECTION ####')

                while round(eps, 6) <= maxCauchyStrain:
                    cauchy_s.append(round(eps, 6))

                    if 'X' in direction:
                        cell[0][0] = (1 + eps) * lx
                    strain_x.append(round(cell[0][0] - lx, 6))
                    strain_per_x.append(round(cell[0][0] / lx, 6))

                    if 'Y' in direction:
                        cell[1][1] = (1 + eps) * ly
                    strain_y.append(round(cell[1][1] - ly, 6))
                    strain_per_y.append(round(cell[1][1] / ly, 6))

                    # apply current step setting and calculator
                    mol.set_cell(cell)
                    mol.center()
                    mol.set_calculator(calc)

                    # Optimize geometry for current cell with GPAW optimization
                    # opt = QuasiNewton(mol, trajectory=f'{out_path}temp.traj',logfile='BFGS.log', )  # enable to write a file with GPAW entire output
                    # opt.run(fmax=fmax, steps=1000)  # run optimization until fmax is reached

                    # instead of usng ASE routines I'll to leaving it to DFTB
                    calc.calculate(mol)
                    MPI.COMM_WORLD.Barrier()

                    # read optimization output after all nodes are done
                    # mol = read(f'{out_path}temp.traj')
                    # en = mol.get_potential_energy()

                    mol = read('geo_end.gen')
                    en = read_energy_dftb()

                    # remove temp.traj file after all nodes got the mol
                    MPI.COMM_WORLD.Barrier()

                    # output
                    traj.write(mol)
                    energy.append(en)

                    MPI.COMM_WORLD.Barrier()

                    parprint('Original length x  |' + str(round(lx, 6)) + ' A')
                    parprint('Original Length y  |' + str(round(ly, 6)) + ' A')
                    parprint('Original Cell Area |' + str(round(lx * ly, 6)) + ' A²')
                    parprint('New Length x       |' + str(round(cell[0][0], 6)) + ' A')
                    parprint('New Length y       |' + str(round(cell[1][1], 6)) + ' A')
                    parprint('Strain x           |' + str(round(strain_x[i], 6)) + ' A')
                    parprint('Strain y           |' + str(round(strain_y[i], 6)) + ' A')
                    parprint('Stretch Ratio x    |' + str(round(strain_per_x[i], 6)) + ' A')
                    parprint('Stretch Ratio y    |' + str(round(strain_per_y[i], 6)) + ' A')
                    parprint('#######################')
                    parprint('Cauchy Strain      |' + str(round(eps, 6)))
                    parprint('Total Energy       |' + str(round(en, 6)) + ' eV')
                    parprint('Energy per Area    |' + str(round(en / (lx * ly), 6)) + ' eV/A²')
                    parprint('\n')
                    i += 1
                    eps += step

                # write output file
                import csv
                import numpy
                data = numpy.array(
                    [['Cell_vector_X'] + [lx] * (i - 1), ['Cell_Vector_Y'] + [ly] * (i - 1), strain_x, strain_y,
                     strain_per_x, strain_per_y, cauchy_s, energy])
                data = data.transpose()
                with open(f'{method}-{calc_base}_{mol_name}_{direction}.csv', 'w+') as out:
                    write = csv.writer(out)
                    write.writerows(data)

                # cleanup
                MPI.COMM_WORLD.Barrier()
                cleanup(out_path)
    os.environ["ASE_DFTB_COMMAND"] = curr_ase_dftb_command

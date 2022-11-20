# This contains the basic class of Cody, used to control all of the performed simulations
# every folder file juggling should be places here to be done and written once.
import os
from mpi4py import MPI


# noinspection PyUnresolvedReferences
class Cody:
    # This method sets required initial variables
    # I'll put the simulation parameters here
    def __init__(self, **kwargs):
        self.method = kwargs.get('method', 'DFTB')
        self.voice = kwargs.get('voice', False)
        self.label = kwargs.get('label', 'dftb_output')
        self.kpts = kwargs.get('kpts', (4, 4, 4))
        self.latticeOpt = kwargs.get('latticeOpt', False)
        self.fixAngles = kwargs.get('fixAngles', False)
        self.fixLengths = kwargs.get('fixLengths', [False, False, False])
        self.maxForce = kwargs.get('maxForce', 1E-4)
        self.maxDriverSteps = kwargs.get('maxDriverSteps', 10000)
        self.SCC = kwargs.get('SCC', True)
        self.maxSCC = kwargs.get('maxSCC', 1E-2)
        self.maxSCCSteps = kwargs.get('maxSCCSteps', 1000)
        self.fermiFilling = kwargs.get('fermiFilling', 0.0)
        self.path = kwargs.get('path', 'Please Supply a Path')
        self.BZ_step = kwargs.get('BZ_step', 1E-2)
        self.interactive_plot = kwargs.get('interactive_plot', False)

        print("\n\nHello, I am Cody!\n\n")
        if self.voice:
            os.system('spd-say "Hello, I am Cody!"')

        # No idea how I'll pass this one consistently between calculators .-.
        # currently I have no clue how I should pass dftb so I'll just disable it
        # and always optimize the whole molecule
        # self.movedAtoms = kwargs.get('movedAtoms', 'All')

    # Check-up Method NOT YET IMPLEMENTED
    def check_parameters(self):
        try:
            assert (len(self.fixLengths) == 3)
        except AssertionError:
            print('"FixLengths" argument should be a list of three Booleans!\n example: [False,False,False]]')
            exit()

    # Generic utility methods
    @staticmethod
    def fetch_molecule_list():
        input_list = os.listdir()
        mol_list = []
        for inputFile in input_list:
            if '.traj' in inputFile:
                mol_list.append(inputFile)
        return mol_list

    @staticmethod
    def clean_files(out_path):
        print('\n##### CLEANUP START ######\n')
        if MPI.COMM_WORLD.Get_rank() == 0:
            warning = True
            output_list = os.listdir()
            for outFile in output_list:
                keep = '.traj' in outFile or '.py' in outFile or os.path.isdir(
                    outFile) or 'FermiLevels.out' == outFile or 'effMass.out' == outFile
                if os.path.isdir(out_path):
                    if warning:
                        print(f'rewriting contents of folder: {out_path}')
                        warning = False
                    if not keep:
                        print(f'{outFile} -> {out_path}{outFile}')
                        os.rename(outFile, out_path + outFile)
                else:
                    os.mkdir(out_path)
                    if warning:
                        print(f'\ncreating new folder: {out_path}')
                        warning = False
                    if not keep:
                        print(f'{outFile} -> {out_path}{outFile}')
                        os.rename(outFile, out_path + outFile)
        print('\n##### CLEANUP DONE #####\n')

    def boolean_to_string(self):
        # convert logical variable into DFTB+ pattern
        if type(self.fixAngles == bool):
            if self.fixAngles:
                self.fixAngles = 'Yes'
            else:
                self.fixAngles = 'No'

        for i in range(len(self.fixLengths)):
            if type(self.fixLengths[i] == bool):
                if self.fixLengths[i]:
                    self.fixLengths[i] = 'Yes  '
                else:
                    self.fixLengths[i] = 'No  '

        if type(self.latticeOpt) == bool:
            if self.latticeOpt:
                self.latticeOpt = 'Yes'
            else:
                self.latticeOpt = 'No'

        if type(self.SCC) == bool:
            if self.SCC:
                self.SCC = 'Yes'
            elif type(self.SCC) != str:
                self.SCC = 'No'

    # Std calculators to be used unless there's a specific setup for the analysis required
    def fetch_dftb_calc(self, cluster):
        from ase.calculators.dftb import Dftb
        self.boolean_to_string()
        eVA_to_HaBohr = 0.01944689673

        if cluster:
            calc = Dftb(label=self.label,
                        Driver_="ConjugateGradient",
                        Driver_MaxForceComponent=self.maxForce * eVA_to_HaBohr,
                        Driver_MaxSteps=self.maxDriverSteps,
                        Driver_MovedAtoms='1:-1',
                        Driver_AppendGeometries='Yes',
                        Hamiltonian_SCC=self.SCC,
                        Hamiltonian_SCCTolerance=self.maxSCC,
                        Hamiltonian_MaxSCCIterations=self.maxSCCSteps,
                        Hamiltonian_Filling=f"Fermi{{Temperature [K] = {self.fermiFilling} }}",
                        )
        else:
            calc = Dftb(label=self.label,
                        kpts=self.kpts,
                        Driver_="ConjugateGradient",
                        Driver_MaxForceComponent=self.maxForce * eVA_to_HaBohr,
                        Driver_MovedAtoms='1:-1',
                        Driver_LatticeOpt=self.latticeOpt,
                        Driver_FixAngles=self.fixAngles,
                        Driver_FixLengths=self.fixLengths[0] + self.fixLengths[1] + self.fixLengths[2],
                        Driver_MaxSteps=self.maxDriverSteps,
                        Driver_AppendGeometries='Yes',
                        Hamiltonian_SCC=self.SCC,
                        Hamiltonian_SCCTolerance=self.maxSCC,
                        Hamiltonian_MaxSCCIterations=self.maxSCCSteps,
                        Hamiltonian_Filling=f"Fermi{{Temperature [K] = {self.fermiFilling} }}",
                        )
        return calc

    def optimize(self):
        from SimLab_OOP.analysis import optimize
        from ase.io import read
        from ase.io import write

        molecules = self.fetch_molecule_list()
        for molecule in molecules:
            mol_name = os.path.splitext(os.path.basename(molecule))[0]
            mol = read(molecule)
            out_path = f'Optimize_{self.method}_{mol_name}' + os.sep

            print(f'{self.method} optimization for {mol_name}')
            pbc = mol.get_pbc()
            if True in pbc:
                print('Some direction has pbc, if "lattice" option is set to True DFTB will '
                      'perform lattice optimization')
                calc = self.fetch_dftb_calc(cluster=False)
            else:
                print('No direction has pbc, DFTB will NOT perform lattice optimization')
                calc = self.fetch_dftb_calc(cluster=True)

            # calculation
            optimize.run(self.method, mol, calc)

            # cleanup
            self.clean_files(out_path)

            # read output and write .traj file after cleanup when I'm sure a folder is there
            mol = read(f'{out_path}geo_end.gen')
            mol.center()
            write(f'{out_path}{self.method}_{mol_name}_end.traj', mol)
            print('\n\n')

        if self.voice:
            os.system('spd-say "It is done"')

    def evaluate_band_structure(self):
        import shutil
        from SimLab_OOP.analysis import bands
        from ase.io import read

        # setup
        molecules = self.fetch_molecule_list()
        for molecule in molecules:
            mol_name = os.path.splitext(os.path.basename(molecule))[0]
            out_path = f'Optimize_{self.method}_{mol_name}' + os.sep
            mol = read(f'{out_path}{self.method}_{mol_name}_end.traj')

            print(f'{self.method} band structure for {mol_name}')
            pbc = mol.get_pbc()
            if True in pbc:
                print('Some direction has pbc, the molecule is valid!')
                shutil.copyfile(f'{out_path}charges.bin', f'{os.getcwd()}{os.sep}charges.bin')

                # calculation
                bands.run(self.method, self.path, self.BZ_step, self.fermiFilling, mol, mol_name)

                # cleanup
                self.clean_files(out_path)
                print('\n\n')
            else:
                print('No direction has pbc, the molecule is NOT valid! ( Yet :o )\n\n')

    def view_band_structure(self):
        from SimLab_OOP.view import bands
        from ase.io import read
        molecules = self.fetch_molecule_list()
        for molecule in molecules:
            mol_name = os.path.splitext(os.path.basename(molecule))[0]
            out_path = f'Optimize_{self.method}_{mol_name}' + os.sep
            mol = read(f'{out_path}{self.method}_{mol_name}_end.traj')

            print(f'Plotting {self.method} band structure for {mol_name}')
            pbc = mol.get_pbc()
            if True in pbc:
                print('Some direction has pbc, the molecule is valid!')
                if self.method == 'DFTB':
                    bands.run_dftb(self.method, out_path, mol_name, mol,
                                   self.path, self.BZ_step, self.interactive_plot)
                    print('\n\n')
            else:
                print('No direction has pbc, the molecule is NOT valid! ( Yet :o )\n\n')

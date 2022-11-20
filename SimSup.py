# This contains the basic class of Cody, used to control all of the performed simulations
# every folder file juggling should be places here to be done and written once.
import os
from mpo4py import MPI


class Cody:
    # This method sets required initial variables
    # I'll put the simulation parameters here
    def __init__(self, **kwargs):
        self.label = kwargs.get('label', 'dftb_output')
        self.kpts = kwargs.get('kpts',(4,4,4))
        self.latticeOpt = kwargs.get('latticeOpt',False)
        self.fixAngles = kwargs.get('fixAngles', False)
        self.fixLengths = kwargs.get('fixLengths', [False, False, False])
        self.cluster = kwargs.get('cluster', False)
        self.maxForce = kwargs.get('maxForce', 1E-4)
        self.maxDriverSteps = kwargs.get('maxDriverSteps', 10000)
        self.SCC = kwargs.get('SCC', True)
        self.maxSCC = kwargs.get('maxSCC',1E-2)
        self.maxSCCSteps = kwargs.get('maxSCCSteps',1000)
        self.fermiFilling = kwargs.get('fermiFilling', 0.0)

        # No idea how I'll pass this one consistently between calculators .-.
        self.movedAtoms = kwargs.get('movedAtoms', 'All')

    # Check-up Methods
    def check_parameters(self):
        try:
            assert (len(self.fixLengths) == 3)
        except AssertionError:
            print('"FixLengths" argument should be a list of three Booleans!\n example: [False,False,False]]')
            exit()

    def boolean_to_string(self):
        # convert logical variable into DFTB+ pattern
        if self.fixAngles:
            self.fixAngles = 'Yes'
        elif type(self.fixAngles) != str:
            self.fixAngles = 'No'

        for i in range(len(self.fixLengths)):
            if self.fixLengths[i]:
                self.fixLengths[i] = 'Yes'
            elif type(self.fixLengths[i]) != str:
                self.fixLengths[i] = 'No'

        if self.latticeOpt:
            self.latticeOpt = 'Yes'
        elif type(self.latticeOpt) != str:
            self.latticeOpt = 'No'

        if self.SCC:
            self.SCC = 'Yes'
        elif type(self.SCC) != str:
            self.SCC = 'No'

    # Methods of Cody
    def setup_dftb(self):
        from ase.calculators.dftb import Dftb
        self.boolen_to_string()
        eVA_to_HaBohr = 0.01944689673
        if self.cluster:
            calc = Dftb(label=self.label,
                        Driver_="ConjugateGradient",
                        Driver_MaxForceComponent=self.maxforce * eVA_to_HaBohr,
                        Driver_MaxSteps=self.maxDriverSteps,
                        Driver_MovedAtoms=self.movedAtoms,
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
                        Driver_MaxForceComponent=self.maxforce * eVA_to_HaBohr,
                        Driver_MovedAtoms=self.movedAtoms,
                        Driver_LatticeOpt=self.lattice,
                        Driver_FixAngles=self.fixAngles,
                        Driver_FixLengths=self.fixLengths,
                        Driver_MaxSteps=self.maxDriverSteps,
                        Driver_AppendGeometries='Yes',
                        Hamiltonian_SCC=self.SCC,
                        Hamiltonian_SCCTolerance=self.maxSCC,
                        Hamiltonian_MaxSCCIterations=self.maxSCCSteps,
                        Hamiltonian_Filling=f"Fermi{{Temperature [K] = {self.fermiFilling} }}",
                        )
        return calc

    @staticmethod
    def clean_files(self):
        print("Cleaning up everything!")

    def optimize(self):
        self.setup_dftb()

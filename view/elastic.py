import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from mpi4py import MPI
import SimLab.utils as utils
import os


########## READING ##########
def readCSV(Path, File):
    # data format being returned data[step][info]
    # step -> various strain applied to the sheet
    # info one of the various tacked informations
    # standard column for info index are:
    # Cell_vector_X,Cell_Vector_Y,StrainX,StrainY,StrainPerX,StrainPerY,CauchyStrain,Energy
    # can be accessed by evaluating data[0]
    import csv
    with open(Path + File, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        cauchy = []
        energy = []
        for data in spamreader:
            try:
                data[0] = float(data[0])
                data[1] = float(data[1])
                data[2] = float(data[2])
                data[3] = float(data[3])
                data[6] = float(data[6])
                data[7] = float(data[7])
                cauchy.append(data[6])
                energy.append(data[7])
            except:
                continue
    if 'XX' in File:
        return 'XX', cauchy, energy, (data[0]) * (data[1])
    elif 'YY' in File:
        return 'YY', cauchy, energy, (data[0]) * (data[1])
    else:
        return 'XY', cauchy, energy, (data[0]) * (data[1])


def run(method, mol_name, out_path, interactive_plot):
    # setup of constant variables for all files and directions
    eV_to_Joule = (1.602 * 10 ** (-19))
    ang_to_meter = 1 * 10 ** (-10)

    # setup
    Constants = []
    fig = plt.figure()

    with open(out_path + 'Elastic_Output.out', 'w+') as log:
        print('\n##### ELASTIC CONSTANT ANALYSIS RESULTS #####\n')
        log.write('##### ELASTIC CONSTANT ANALYSIS RESULTS #####\n')
    folder_list = os.listdir(out_path)

    for folder_file in folder_list:
        if '.csv' in folder_file:
            # read import info from analysis file
            Direction, Cauchy, Energy, Area = readCSV(Path=out_path, File=folder_file)

            # convert energy from eV to J/m² - ground
            ground = min(Energy)
            for i in range(len(Energy)):
                Energy[i] = ((Energy[i] - ground) / Area) * (eV_to_Joule) / (ang_to_meter ** 2)

            # fit parabola to figure
            plt.plot(Cauchy, Energy, 'o', label=Direction)
            c = np.polyfit(Cauchy, Energy, 2)
            Constants.append([Direction, c])
            xFit = np.linspace(Cauchy[0], Cauchy[-1])
            plt.plot(xFit, c[0] * xFit ** 2 + c[1] * xFit + c[2], '-')

    # elastic constants
    Axx = .0
    Ayy = .0
    Axy = .0
    for i in range(len(Constants)):
        if 'XX' in Constants[i][0]:
            Axx = Constants[i][1]
        elif 'YY' in Constants[i][0]:
            Ayy = Constants[i][1]
        else:
            Axy = Constants[i][1]
    utils.parwrite(f'Direction XX Parabolic fit constants: {Axx}', f'{out_path}Elastic_Output.out')
    utils.parwrite(f'Direction YY Parabolic fit constants: {Ayy}', f'{out_path}Elastic_Output.out')
    utils.parwrite(f'Direction XY Parabolic fit constants: {Axy}', f'{out_path}Elastic_Output.out')

    m = 3.34 * 10 ** (-10)  # uncoment for 3D, only applicable to graphene :(
    # m=1 #uncoment for 2D, applicable for everything 2D :)
    C11 = 2 * Axx[0]
    C22 = 2 * Ayy[0]
    C12 = (Axy[0]) - 0.5 * (C11 + C22)
    C11_m = 2 * Axx[0]
    C22_m = 2 * Ayy[0]
    C12_m = Axy[0] - 0.5 * (C11 + C22)

    utils.parwrite(f'C11: {C11:e} J/m²', out_path + 'Elastic_Output.out')
    utils.parwrite(f'C22: {C22:e} J/m²', out_path + 'Elastic_Output.out')
    utils.parwrite(f'C12: {C12:e} J/m²', out_path + 'Elastic_Output.out')

    Ex = ((C11 * C22 - C12 ** 2) / C22)
    Ey = ((C11 * C22 - C12 ** 2) / C11)
    Yxy = (C12 / C11)
    Yyx = (C12 / C22)
    Ex_m = Ex / m
    Ey_m = Ey / m
    Yxy_m = (C12_m / C11_m)
    Yyx_m = (C12_m / C22_m)

    utils.parwrite(f'Ex:  {Ex:e} J/m²   | {Ex_m:e} J/m³', out_path + 'Elastic_Output.out')
    utils.parwrite(f'Ey:  {Ey:e} J/m²   | {Ey_m:e} J/m³', out_path + 'Elastic_Output.out')
    utils.parwrite(f'Yxy: {Yxy:e}        | {Yxy_m:e}', out_path + 'Elastic_Output.out')
    utils.parwrite(f'Yyx: {Yyx:e}        | {Yyx_m:e}', out_path + 'Elastic_Output.out')

    # after analysis is done, set graph aesthetics only once
    plt.legend()
    plt.xlabel('Cauchy Strain [a.u.]')
    plt.ylabel(f'Energy[J/m²]')
    fig.suptitle(f'{method}-{mol_name} Strain Analysis')
    plt.savefig(f'{out_path}{method}_parabolic.png')
    if interactive_plot:
        plt.show()
    else:
        plt.clf()

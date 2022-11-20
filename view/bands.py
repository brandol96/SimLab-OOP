import os

import matplotlib.pyplot as plt
import numpy as np
from SimLab.utils import cleanup
from SimLab.utils import path_dftb
from SimLab.utils import parprint
from ase.io import read
from mpi4py import MPI

# reading functions
plt.rcParams["font.family"] = "serif"


def read_dos(path, mol_name):
    ene = []
    dos = []
    i = 0
    print('read DOS')
    with open(f'{path}{mol_name}.dos.dat') as file:
        for line in file:
            data = line.split()
            ene.insert(i, float(data[0]))
            dos.insert(i, float(data[1]))
            i += 1
    return ene, dos


def read_fermi_levels(path, mol_name):
    print('read Fermi Level')
    lumo = [0, 0, 1000]
    homo = [0, 0, -1000]
    lumo_cur = [0, 0, 0]
    homo_cur = [0, 0, 0]
    with open(f'{path}band.out') as inputFile:
        for line in inputFile:
            txt = line.split()
            if txt != [] and txt[0] == 'KPT':  # true if we have header
                kpt = int(txt[1])  # useful for indirect gap?
            # true if NOT a header and not found lumo
            if txt != [] and txt[0] != 'KPT' and lumo_cur == [0, 0, 0]:
                if float(txt[2]) == 0.0:  # found lumo of current KPOINT
                    lumo_cur = [kpt, float(txt[0]), float(txt[1])]
                else:  # if occupation is not 0.0 then fill homo_cur
                    homo_cur = [kpt, float(txt[0]), float(txt[1])]
            # true we have reached the end of a KPOINT info
            if not txt:  # empty list is a false!
                if lumo[2] >= lumo_cur[2]:
                    lumo = lumo_cur
                if homo[2] <= homo_cur[2]:
                    homo = homo_cur
                lumo_cur = [0, 0, 0]
                homo_cur = [0, 0, 0]
    fermi_e = round((lumo[2] + homo[2]) / 2, 6)
    gap = round(lumo[2] - homo[2], 6)
    print(f'{"molName":<15} {"homo[kpt, Band, eV]":>20} {"lumo[kpt, Band, eV]":>20} {"gap":<3} {"fermi_e":<6}')
    homo_string = f'[{homo[0]},{homo[1]},{homo[2]}]'
    lumo_string = f'[{lumo[0]},{lumo[1]},{lumo[2]}]'
    print(f'{mol_name:<15} {homo_string:>20} {lumo_string:>20} {gap:<3} {fermi_e:<3}')
    return homo, lumo, gap, fermi_e


def output_fermi_levels(mol_name, homo, lumo, fermi_energy, gap):
    try:
        with open('FermiLevels.out', 'r') as inputFile:
            with open('FermiLevels.tmp', 'w+') as tmp:
                for line in inputFile:
                    # print(line)
                    tmp.write(line)
                next_line_list = [mol_name, str(homo), str(lumo), str(fermi_energy), str(gap)]
                next_line = '{: <50} {: >24} {: >24} {: >20} {: >20}'.format(*next_line_list)
                tmp.write(next_line + '\n')
        os.rename('FermiLevels.tmp', 'FermiLevels.out')
    except FileNotFoundError:
        with open('FermiLevels.out', 'w+') as inputFile:
            next_line_list = ['Molecule', 'HOMO', 'LUMO', 'FermiE', 'gap']
            next_line = '{: <50} {: >24} {: >24} {: >20} {: >20}'.format(*next_line_list)
            inputFile.write(next_line + '\n')
            next_line_list = [mol_name, str(homo), str(lumo), str(fermi_energy), str(gap)]
            next_line = '{: <50} {: >24} {: >24} {: >20} {: >20}'.format(*next_line_list)
            inputFile.write(next_line + '\n')


def run_dftb(method, out_path, mol_name, mol, path, step, interactive_plot):
    if method == 'DFTB':
        # load relevant data
        band_data = np.genfromtxt(f'{out_path}{mol_name}.band_tot.dat')
        # contains nXY plot of the band structure, it is a matrix of the form
        # [[kpt1 E_band1 E_band2 ... E_bandM]
        # [kpt2 E_band1 E_band2 ... E_bandM]
        # ...
        # [kptN E_band1 E_band2 ... E_bandM]]

        ref_kpts = path_dftb(path, step, mol, False, False)
        dic_kpts = path_dftb(path, step, mol, False, True)

        ene, dos = read_dos(out_path, mol_name)
        # two lists containing plot data for density of states

        homo, lumo, gap, fermi_e = read_fermi_levels(out_path, mol_name)
        # homo -> info about found homo [KPT,BAND,EV]
        # lumo -> info about found lumo [KPT,BAND,EV]
        # gap  -> gap value calculated for current file
        # fermi_e -> fermi energy calculated for current file

        output_fermi_levels(mol_name, homo, lumo, fermi_e, gap)

        # center plot into fermi energy
        for i in range(len(band_data)):
            band_data[i][1:] = band_data[i][1:] - fermi_e
        for i in range(len(dos)):
            ene[i] = ene[i] - fermi_e

        fermi_e = 0
        # onto the plot itself!

        # a few parameters
        zoom = 6
        title_font = 20
        label_font = 16
        text_font = 14

        print('\n\nstart plot\n\n')

        # setup figure
        fig = plt.figure(1, figsize=(8, 10))  # start a figure
        fig.suptitle(mol_name.replace("-", " "), fontsize=title_font)

        # bands axes
        ax = fig.add_axes([.12, .07, .67, .85])  # axes [left, bottom, width, height]
        ax.set_xticks([])
        ax.set_ylabel('$E - E_f$ (eV)', fontsize=label_font)
        ax.set_ylim([fermi_e - zoom, fermi_e + zoom])

        # dos axes
        dosax = fig.add_axes([.8, .07, .17, .85])  # axes [left, bottom, width, height]
        dosax.fill_between(dos, ene)
        dosax.set_yticks([])
        dosax.set_xticks([])
        dosax.set_xlabel("DOS", fontsize=label_font)
        dosax.set_ylim([fermi_e - zoom, fermi_e + zoom])

        # plot bands and dos
        for i in range(band_data.shape[1] - 1):
            ax.plot(band_data[:, 0], band_data[:, i + 1])
        dosax.plot(dos, ene, color='black')

        # plot Fermi Level on Bands figure
        ax.plot(band_data[:, 0], [fermi_e] * len(band_data), '--', color='black')
        ax.text(5.0, fermi_e + zoom / 50, 'Gap: ' + str(round(gap, 2)) + ' eV', fontsize=text_font)

        # plot vertical lines indicating the Path taken
        # path = details['BZ_path']
        kpoints = []
        kpoints2 = ['\u0393', '\u0393', 'M', 'K', '\u0393', '\u0393']
        kposition = []

        # find Y-points to plot de KLines of the greater and lesses energy valueson Y-axis
        ymin = 0.0
        ymax = 0.0
        for i in range(len(band_data[0, 1:])):
            if min(band_data[:, i + 1]) < ymin:
                ymin = min(band_data[:, i + 1])
            if max(band_data[:, i + 1]) > ymax:
                ymax = max(band_data[:, i + 1])

        # convert DFTB input into a usable list format
        # input string -> [[kpt1,x,y,z], [kpt2,x,y,z], ..., [kptsN,x,y,z]]
        ref_kpts = ref_kpts.split('\n')
        ref_kpts = [x.split() for x in ref_kpts]
        aux = []
        for point in ref_kpts:
            if not point:
                continue
            else:
                point = [float(x) for x in point]
            aux.append(point)
        ref_kpts = aux.copy()

        # plot the vertical Klines
        x = 0
        print('\n search for path:')
        for point in ref_kpts:
            # dftb input states how far one line is from the previous in number of steps
            x += point[0]
            ax.plot([x, x], [ymin, ymax], color='grey')

            # to find the correct kpoint symbol I'll compare the dftb input coordinates to
            coords = point[1:]
            value = [i for i in dic_kpts if dic_kpts[i] == coords]
            print("key by value:", value[0])
            ax.text(x, fermi_e - zoom - text_font / 33, value[0], fontsize=text_font)
        print(ref_kpts)
        print(dic_kpts)

        for i in range(len(kpoints)):
            ax.plot([kposition[i], kposition[i]], [less, most], color='grey')
            ax.text(kposition[i] - 1.5, -5.25, kpoints2[i], fontsize=12)

        # zoom into specif parts of the plot
        fig.savefig(f'{out_path}{method}_{mol_name}_Bands.png')
        if interactive_plot:
            plt.show()
        else:
            plt.clf()
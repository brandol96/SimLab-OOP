import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from mpi4py import MPI
import SimLab.utils as utils
from SimLab.utils import cleanup
from SimLab_OOP.view import bands
import statistics
import os


def output_eff_mass(mol_name, homo, lumo, gap, mass_hole, mass_ele):
    try:
        with open('effMass.out', 'r') as inputFile:
            with open('effMass.tmp', 'w+') as tmp:
                for line in inputFile:
                    # print(line)
                    tmp.write(line)
                next_line_list = [mol_name, str(homo), str(lumo), str(gap), str(mass_hole), str(mass_ele)]
                next_line = '{: <50} {: >24} {: >24} {: >20} {: >20} {: >20}'.format(*next_line_list)
                tmp.write(next_line + '\n')
        os.rename('effMass.tmp', 'effMass.out')
    except FileNotFoundError:
        with open('effMass.out', 'w+') as inputFile:
            next_line_list = ['Molecule', 'HOMO', 'LUMO', 'gap', 'mass_hole', 'mass_elec']
            next_line = '{: <50} {: >24} {: >24} {: >20} {: >20} {: >20}'.format(*next_line_list)
            inputFile.write(next_line + '\n')
            next_line_list = [mol_name, str(homo), str(lumo), str(gap), str(mass_hole), str(mass_ele)]
            next_line = '{: <50} {: >24} {: >24} {: >20} {: >20} {: >20}'.format(*next_line_list)
            inputFile.write(next_line + '\n')


# find indexes of the regions to be fit
# normally dftb+ min are plateaus of energy, to avoid a fit on one side of
# the parabolic region, I search for the mean index of the plateau
# It is possible there are multiple plateaus, mainly if HOMO/LUMO
# are near the edges of the band. I'll consider only the first plateau
def find_homo(y_homo):
    # aux_idx: a list of the indexes of maximum values of y_homo in appearance order
    aux_idx = [i for i, x in enumerate(y_homo) if x == max(y_homo)]
    # if there's a single homo point just go back
    if len(aux_idx) == 1:
        return aux_idx[0]

    # otherwise continue with the search
    for i in range(len(aux_idx)):
        if i + 1 == len(aux_idx):
            homo_idx = int(statistics.mean(aux_idx[0:i]))
            return homo_idx
        if aux_idx[i + 1] != aux_idx[i] + 1:
            homo_idx = int(statistics.mean(aux_idx[0:i]))
            return homo_idx


def find_lumo(y_lumo):
    # aux_idx: a list of the indexes of minimum values of y_lumo in appearance order
    aux_idx = [i for i, x in enumerate(y_lumo) if x == min(y_lumo)]

    print(aux_idx)

    # if there's a single lumo point just go back
    if len(aux_idx) == 1:
        return aux_idx[0]

    # otherwise continue with the search
    for i in range(len(aux_idx)):
        if i + 1 == len(aux_idx):
            lumo_idx = int(statistics.mean(aux_idx[0:i]))
            return lumo_idx
        if aux_idx[i + 1] != aux_idx[i] + 1:
            lumo_idx = int(statistics.mean(aux_idx[0:i]))
            return lumo_idx


def run(mol, mol_name, out_path, BZ_step, interactive_plot):
    # load relevant data
    band_data = np.genfromtxt(f'{out_path}{mol_name}.band_tot.dat')
    homo, lumo, gap, fermi_e = bands.read_fermi_levels(out_path, mol_name)

    # variables
    x = []
    idx_homo = int(homo[1])
    idx_lumo = int(lumo[1])
    y_homo = []
    y_lumo = []
    n = 0

    # conversion constants for multiplication!
    eV_to_Joule = 1.60218E-19
    Joule_to_eV = 1 / 1.60218E-19

    # get HOMO and LUMO bands
    if idx_lumo == idx_homo:
        print('half-filled band')
        for point in band_data:
            x.append(n * BZ_step * 2 * np.pi * 1E10)
            y_homo.append(point[idx_homo] * eV_to_Joule)
            y_lumo.append(point[idx_lumo + 1] * eV_to_Joule)
            n += 1
    else:
        print('total filled band')
        for point in band_data:
            x.append(n * BZ_step * 2 * np.pi * 1E10)
            y_homo.append(point[idx_homo] * eV_to_Joule)
            y_lumo.append(point[idx_lumo] * eV_to_Joule)
            n += 1

    print(max(y_homo))
    print(min(y_lumo))

    homo = [find_homo(y_homo), float(idx_homo + 1), round(max(y_homo) * Joule_to_eV, 3)]
    lumo = [find_lumo(y_lumo), float(idx_lumo + 1), round(min(y_lumo) * Joule_to_eV, 3)]

    # output basic bands
    print(f'Read homo: {homo}')
    print(f'Read lumo: {lumo}')
    plt.plot(x, y_homo, 'o', label='homo', markersize=2, zorder=5)
    plt.plot(x, y_lumo, 'o', label='lumo', markersize=2, zorder=5)

    # fit parameters and plot homo and lumo bands
    npoints = int(0.05 * len(x))

    lumo_idx = lumo[0]
    homo_idx = homo[0]

    # if we're lacking left points extend the band

    # I'm having trouble: when the band is extended, the plateau may be waaaay
    # too large, thus the fit centering teh fit region still falls outside the
    # enlarged band, I'll add a "safety factor" to enalarge the band way beyond
    # what is actually needed, this WILL affect performance, because the way
    # I'm searching for min and max goes through the entire list, but I can't think
    # of a solution right now :(
    safety = 3 * npoints
    if lumo_idx - npoints <= 0 or homo_idx - npoints <= 0:
        x = np.arange(-safety, len(x) + 1, 1)
        x = [d * BZ_step * 2 * np.pi * 1E10 for d in x]
        y_homo = y_homo[len(y_homo) - safety - 1: len(y_homo)] + y_homo
        y_lumo = y_lumo[len(y_lumo) - safety - 1: len(y_homo)] + y_lumo
        homo = [find_homo(y_homo), float(idx_homo + 1), round(max(y_homo) * Joule_to_eV, 3)]
        lumo = [find_lumo(y_lumo), float(idx_lumo + 1), round(min(y_lumo) * Joule_to_eV, 3)]
        print('\nHOMO was extended')
        print(f'New homo: {homo}')
        print(f'New lumo: {lumo}')

    # if we're lacking right points extend the band
    if lumo_idx + npoints >= len(x) or homo_idx + npoints >= len(x):
        x = np.arange(0, len(x) + safety + 1, 1)
        x = [d * BZ_step_size * 2 * np.pi * 1E10 for d in x]
        y_homo = y_homo + y_homo[0:safety + 1]
        y_lumo = y_lumo + y_lumo[0:safety + 1]
        homo = [find_homo(y_homo), float(idx_homo + 1), round(max(y_homo) * Joule_to_eV, 3)]
        lumo = [find_lumo(y_lumo), float(idx_lumo + 1), round(min(y_lumo) * Joule_to_eV, 3)]
        print('\nLUMO was extended')
        print(f'New homo: {homo}')
        print(f'New lumo: {lumo}')

    lumo_idx = lumo[0]
    homo_idx = homo[0]

    y_fit_lumo = y_lumo[lumo_idx - npoints:lumo_idx + npoints]
    y_fit_homo = y_homo[homo_idx - npoints:homo_idx + npoints]
    x_fit_homo = x[homo_idx - npoints:homo_idx + npoints]
    x_fit_lumo = x[lumo_idx - npoints:lumo_idx + npoints]

    # plot fit regions
    plt.plot(x, y_homo, color='grey', zorder=0)
    plt.plot(x, y_lumo, color='grey', zorder=0)

    print(f'HOMO fit region: {homo_idx - npoints}    {homo_idx + npoints}')
    print(f'LUMO fit region: {homo_idx - npoints}    {homo_idx + npoints}')

    plt.plot(x_fit_homo, y_fit_homo, 'o', label='fit region', markersize=2, color='black', zorder=10)
    plt.plot(x_fit_lumo, y_fit_lumo, 'o', markersize=2, color='black', zorder=10)

    # calculate parabolic fit
    fit_lumo = np.polyfit(x_fit_lumo, y_fit_lumo, 2)
    fit_homo = np.polyfit(x_fit_homo, y_fit_homo, 2)

    # plot fit
    for i in range(len(x_fit_homo)):
        y_fit_lumo[i] = fit_lumo[0] * x_fit_lumo[i] ** 2 + fit_lumo[1] * x_fit_lumo[i] + fit_lumo[2]
        y_fit_homo[i] = fit_homo[0] * x_fit_homo[i] ** 2 + fit_homo[1] * x_fit_homo[i] + fit_homo[2]
    plt.plot(x_fit_lumo, y_fit_lumo, label=f'fit constant a = {fit_lumo[0]}', zorder=15)
    plt.plot(x_fit_homo, y_fit_homo, label=f'fit constant a = {fit_homo[0]}', zorder=15)

    # effective mass
    # hbar  = 6.5821E-16
    hbar = 1.054571817E-34
    eMass = 9.10938E-31

    Effective_constant = (hbar ** 2) / (2 * eMass)
    eff_el = round(Effective_constant / fit_lumo[0], 2)
    eff_hol = round(-Effective_constant / fit_homo[0], 2)
    gap = round(lumo[2] - homo[2], 3)

    # output final results
    print(f'points considered as HOMO and LUMO: {homo}   |   {lumo}')
    print(f'gap of considered points: {gap}')
    print(f'number of points in fit region: {npoints}')
    print(f'\n\nEffective Mass constant: {Effective_constant}')
    print(f'Hole fit: {fit_homo[0]}')
    print(f'Electron fit: {fit_lumo[0]}')
    print('\n##########\n')
    print(f'Electron Effective Mass: {eff_el}')
    print(f'Hole Effective Mass: {eff_hol}')
    print('\n\n')
    output_eff_mass(mol_name, homo, lumo, gap, eff_hol, eff_el)

    plt.title(f'{mol_name}: el mass = {eff_el} | hl mass = {eff_hol}')
    plt.xlabel(f'Distance [1/m]')
    plt.ylabel(f'Enegy [J]')
    plt.legend()

    plt.savefig(f'{out_path}{mol_name}_EffMassFit.png')
    if interactive_plot:
        plt.show()
    else:
        plt.clf()
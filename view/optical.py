import os

import numpy
from scipy import constants

import matplotlib.pyplot as plt
import numpy as np
from SimLab.utils import cleanup
from ase.io import read
from mpi4py import MPI


def read_spec_ev(path):
    x = []
    y = []
    with open(f'{path}spec-ev.dat') as inp:
        for line in inp:
            data = line.split()
            x.append(float(data[0]))
            y.append(float(data[1]))
    return x, y


def run(method, out_path, mol_name, interactive_plot, directions, laser, fourrierDamp, fieldStrenght):
    # a few parameters
    zoom = 6
    title_font = 20
    label_font = 16
    text_font = 14

    print('\n\nstart plot\n\n')
    # setup figure
    fig = plt.figure()  # start a figure
    fig.suptitle(mol_name.replace("_", " "), fontsize=title_font)
    total = np.array([])
    d = 0.
    if 'X' in directions:
        print('\nLoading X direction...')
        mu = np.loadtxt(f'{out_path}mux.dat')  # response to excitation in x direction
        if total.size == 0:
            total = (mu[:, 1] - mu[0, 1])
        else:
            total += (mu[:, 1] - mu[0, 1])
        d += 1
    if 'Y' in directions:
        print('\nLoading Y direction...')
        mu = np.loadtxt(f'{out_path}muy.dat')  # response to excitation in y direction
        if total.size == 0:
            total = (mu[:, 2] - mu[0, 2])
        else:
            total += (mu[:, 2] - mu[0, 2])
        d += 1
    if 'Z' in directions:
        print('\nLoading Z direction...')
        mu = np.loadtxt(f'{out_path}muz.dat')  # response to excitation in z direction
        if total.size == 0:
            total = (mu[:, 2] - mu[0, 2])
        else:
            total += (mu[:, 3] - mu[0, 3])
        d += 1
    if laser:
        mu = np.loadtxt(f'{out_path}mu.dat')

    average = total / d
    damp = np.exp(-mu[:, 0] / fourrierDamp)
    field = fieldStrenght

    spec = np.fft.rfft(damp * average, 10 * mu.shape[0])
    hplanck = constants.physical_constants['Planck constant in eV s'][0] * 1.0E15
    cspeednm = constants.speed_of_light * 1.0e9 / 1.0e15
    energsev = np.fft.rfftfreq(10 * mu.shape[0], mu[1, 0] - mu[0, 0]) * hplanck
    frec = np.fft.rfftfreq(10 * mu.shape[0], (mu[1, 0] - mu[0, 0]) * 1.0E-15)
    absorption = -2.0 * energsev * spec.imag / np.pi / field
    energsnm = constants.nu2lambda(frec[1:]) * 1.0E9

    emin = 0.5
    emax = 30.
    wvlmin = hplanck * cspeednm / emax
    wvlmax = hplanck * cspeednm / emin

    np.savetxt(f'{out_path}spec-ev.dat', np.column_stack((energsev[(energsev > emin) & (energsev < emax)], \
                                                          absorption[(energsev > emin) & (energsev < emax)])))
    np.savetxt(f'{out_path}spec-nm.dat', np.column_stack((energsnm[(energsnm > wvlmin) & (energsnm < wvlmax)], \
                                                          absorption[1:][(energsnm > wvlmin) & (energsnm < wvlmax)])))

    X, Y = read_spec_ev(out_path)

    plt.plot(X, Y)
    fig.savefig(f'{out_path}{method}_{mol_name}_Optial.png')
    if interactive_plot:
        plt.show()
    else:
        plt.clf()
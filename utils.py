import os
import numpy as np
from mpi4py import MPI


# This script contains some pertinent test of convergence of the methods implement in SimLab
# TODO: create 'output file' function to remove this task from specific analysis
# TODO: remodel 'cleanup' functon to receive a list of filenames to be ignored

def parprint(text):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(text)


def parwrite(text, outfile):
    try:
        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(outfile, 'r') as outFile:
                with open('temp', 'w+') as temp:
                    for line in outFile:
                        temp.write(line)
                    print(text)
                    temp.write(text + '\n')
            os.rename('temp', outfile)
    except FileNotFoundError:
        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(outfile, 'w+') as outFile:
                outFile.write(text + '\n')


def read_energy_dftb():
    with open("detailed.out") as output:
        for line in output:
            if line != '\n':
                data = line.split()
            if data[0] == 'Total' and data[1] == 'energy:':
                energy = float(data[4])
    # print('readEnergy_dftb returning energy: '+str(energy))
    return energy


def path_dftb(path, dK, mol, verbose, get_dict):
    path2kpts = {'G': [0.0, 0.0, 0.0],
                 'M': [0.5, 0.0, 0.0],
                 'K': [2 / 3, 1 / 3, 0.0]}

    if get_dict: return path2kpts
    dftb_path = ''
    cell = mol.get_cell()
    reci = cell.reciprocal()
    k_f = path2kpts[path[0]]
    i = j = 0
    for point in path:
        k_i = k_f
        k_f = path2kpts[point]
        k_c = k_f[0] * reci[0] + k_f[1] * reci[1] + k_f[2] * reci[2]
        k_o = k_i[0] * reci[0] + k_i[1] * reci[1] + k_i[2] * reci[2]
        length = (k_c[0] - k_o[0]) ** 2 + (k_c[1] - k_o[1]) ** 2 + (k_c[2] - k_o[2]) ** 2
        length = np.sqrt(length)
        n = int(length / dK)
        if verbose:
            parprint(f'length from {path[j]} to {path[i]} is: {length}')
        if verbose:
            parprint(f'{n} steps of {dK} will add up to {n * dK}\n')
        if n == 0:
            n = 1
        dftb_path += f'{n}    {k_f[0]}    {k_f[1]}    {k_f[2]}\n'
        j = i
        i += 1
    return dftb_path

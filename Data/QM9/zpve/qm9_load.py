from ase.io import read, write, Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.visualize import view
from ase import Atoms
from ase.units import Hartree,eV
import numpy as np
import random
import os
import re
from datetime import datetime

def date(fmt="%m/%d/%Y %H:%M:%S"):
    return datetime.now().strftime(fmt)

def get_symbols(symbol_list):
    num = 1
    symbols = ''
    for i in range(len(symbol_list)-1):
        if symbol_list[i] == symbol_list[i+1]:
            num += 1
        else:
            symbols += symbol_list[i-num+1] + f'{num}'
            num = 1
    symbols += symbol_list[i-num+2] + f'{num}'

    return symbols


def get_atoms(xyz):
    num_of_atoms = int(xyz[0])
    symbol_list = []
    positions = []
    for i in range(num_of_atoms):
        line = xyz[2+i].split()
        for l in range(len(line)):
            if '*' in line[l]:
                line[l] = line[l].replace('*^','e')
        symbol_list.append(line[0])
        positions.append([float(line[1]), float(line[2]), float(line[3])])
    atoms = Atoms(get_symbols(symbol_list), np.array(positions))

    return atoms


def load_qm9_data(data_path, excepted_list):
    files_with_num = []
    pattern = re.compile(r'dsgdb9nsd_(\d+)\.xyz')

    for file in os.listdir(data_path):
        match = pattern.match(file)
        if match:
            number = int(match.group(1))
            files_with_num.append((file, number))
    print('\n < Number of data >')
    print(f' - Total    :  {len(files_with_num)}')

    files_with_num.sort(key=lambda x: x[1])
    filtered_files = [file for file, number in files_with_num \
                                if number not in excepted_list]

    return filtered_files

def employ_cell(atoms, lattice_params=None, center=True):
    if lattice_params == None:
        lattice_params = [10, 10, 10]
    if center:
        pos = atoms.get_positions()
        cell = np.diag(lattice_params)
        cm = atoms.get_center_of_mass()
        pos[:,0] -= (cm[0] - cell[0][0]/2)
        pos[:,1] -= (cm[1] - cell[1][1]/2)
        pos[:,2] -= (cm[2] - cell[2][2]/2)
        atoms.set_positions(pos)
        atoms.set_cell(cell)
    return atoms



print(date())
seed = 42
ntrain = 110000
nvalid = 10000
# ntest = total - ntrain - nvalid - nexcepted

EXCEPT_DATA = open('uncharacterized.txt')
except_data = EXCEPT_DATA.readlines()
except_indices = []
for i in range(len(except_data)):
    try:
        except_index = int(except_data[i].split()[0])
        except_indices.append(except_index)
    except:
        pass

data_path = 'maindata/'
qm9_xyz_files = load_qm9_data(data_path, except_indices)
random.seed(seed)
random.shuffle(qm9_xyz_files)
ntest = len(qm9_xyz_files) - ntrain - nvalid
print(f' - Excluded :  {len(except_indices)}')

split = {'train': ntrain, 'valid': nvalid, 'test':ntest}
acc = 0
num_atoms = []
hartree_to_ev = Hartree * eV  # 1 Hartree = 27.211386024367243 eV

for key in split:
    num = 0
    traj = Trajectory(f'qm9_{key}.traj', 'w')
    for i in range(split[key]):
        XYZ = open(f'maindata/{qm9_xyz_files[acc+i]}')
        xyz = XYZ.readlines()
        atoms = get_atoms(xyz)
        atoms = employ_cell(atoms, [10, 10, 10])
        atoms.pbc = False
        num_atoms.append(len(atoms))

        energy_hartree = float(xyz[1].split()[11])
        energy_ev = energy_hartree * hartree_to_ev

        calc = SinglePointCalculator(atoms, energy=energy_ev)
        atoms.set_calculator(calc)
        traj.write(atoms)
        num += 1
    print(f' - {key:8} :  {num}')
    print(date())
    acc += num
    traj.close()
print(f' - mean of num(atoms) :  {np.mean(num_atoms)}')
print(f' - max of num(atoms) :  {np.max(num_atoms)}')
print(' ')

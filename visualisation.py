import matplotlib.pyplot as plt
import numpy as np
import json 

def get_bonds(A, r):
    bonds = []
    for i in range(0, len(A)):
        for j in range(0, i):
            if A[i][j][0] > 0:
              bonds.append([r[i][:3], r[j][:3]])
    bonds = np.array(bonds).transpose((0, 2, 1))
    return bonds

def plot_sample(sample, colours=['red', 'blue', 'green']): #makes a 3d plot of the system
    reactants = sample['reactants']
    products = sample['products']
    residues = sample['residues']

    reactants_r = []
    products_r = []
    residues_r = []

    for vec in reactants['F']:
        reactants_r.append(vec[1:4])

    reactant_bonds = get_bonds(reactants['A'], reactants_r)
    
    for vec in products['F']:
        products_r.append(vec[1:4])

    product_bonds = get_bonds(products['A'], products_r)

    for atoms in residues:
        residues_r.append(atoms[3:6])

    # residue_bonds = get_bonds(residues['A'], residues_r)



    reactants_r = np.array(reactants_r).T
    products_r = np.array(products_r).T
    residues_r = np.array(residues_r).T

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.scatter(reactants_r[0], reactants_r[1], reactants_r[2], c=colours[0])
    for bond in reactant_bonds:
        ax1.plot(bond[0], bond[1], bond[2], color='black')
    ax2.scatter(products_r[0], products_r[1], products_r[2], c=colours[1])
    for bond in product_bonds:
        ax2.plot(bond[0], bond[1], bond[2], color='black')
    ax3.scatter(residues_r[0], residues_r[1], residues_r[2], c=colours[2])
    # for bond in residue_bonds:
    #     ax3.plot(bond[0], bond[1], bond[2])
    plt.show()

def plot_mechanism_step(filename='macie_entry_1_1_1.mrv'):
    with open(filename, 'r') as file:
        lines = file.readlines()[0].split('>')

    done = False
    atoms = {}
    for line in lines:
        if line[1:5] == 'atom':
            pass
    return


if __name__ =='__main__':
    with open('datasets/X_processed_dataset.json') as f:
        test = json.load(f)

    plot_sample(test['X']["58"])
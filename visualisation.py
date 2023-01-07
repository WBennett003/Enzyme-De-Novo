import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json 

def plot_dataset(sample):
    reactants = sample['reactants']
    products = sample['products']
    residues = sample['residues']

    ax, fig = plt.subplots(3)

    sns.heatmap(reactants['A'][0, :, :, 0], ax=ax)
    sns.heatmap(products['A'][0, :, :, 0], ax=ax)

    plt.show()    


def get_bonds(A, r):
    bonds = []
    for i in range(0, len(A)):
        for j in range(0, i):
            if A[i][j][0] > 0:
              bonds.append([r[i], r[j]])
    bonds = np.array(bonds).transpose((0, 2, 1))#.tolist()
    return bonds

def get_bonds_and_positions(F, A):
    compound_r = []
    compound_bonds = []
    for i,f in enumerate(F): #addding As together
        temp_position = []
        for vec in f:
            temp_position.append(vec[1:4])

        compound_r.extend(temp_position)
        compound_bonds.extend(get_bonds(A[i], temp_position))
    return compound_r, compound_bonds

def plot_sample(sample, colours=['red', 'blue', 'green']): #makes a 3d plot of the system
    reactants = sample['reactants']
    products = sample['products']
    residues = sample['residues']

    
    reactants_r, reactant_bonds = get_bonds_and_positions(reactants['F'], reactants['A'])
    

    
    products_r, product_bonds = get_bonds_and_positions(products['F'], products['A'])
  
    
    theo = []
    for pdb in residues:
        theozymes = []
        for atoms in residues[pdb]['F']:
            theozymes.append(atoms[3:6])

        b = get_bonds(residues[pdb]['A'], theozymes)
        theo.append([theozymes, b])
    theo = theo[0]
    residues_r = theo[0]
    residue_bonds = theo[1]

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
    for bond in residue_bonds:
        ax3.plot(bond[0], bond[1], bond[2], color='black')
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
    
    q = ''
    while q != 'Quit':
        q = input("id : ")
        plot_sample(test['X'][q])
        plot_dataset(test['X'][q])

import matplotlib.pyplot as plt
import numpy as np
import json 

def plot_sample(sample, colours=['red', 'blue', 'green']): #makes a 3d plot of the system
    reactants = sample['reactants']
    products = sample['products']
    residues = sample['residues']

    reactants_r = []
    products_r = []
    residues_r = []

    for vec in reactants['F']:
        reactants_r.append(vec[1:5])

    for vec in products['F']:
        products_r.append(vec[1:5])

    for atoms in residues:
        residues_r.append(atoms[3:7])

    reactants_r = np.array(reactants_r).T
    products_r = np.array(products_r).T
    residues_r = np.array(residues_r).T

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(reactants_r[0], reactants_r[1], reactants_r[2], c=colours[0])
    ax.scatter(products_r[0], products_r[1], products_r[2], c=colours[1])
    ax.scatter(residues_r[0], residues_r[1], residues_r[2], c=colours[2])
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
    
    plot_sample(test['X']["1"])
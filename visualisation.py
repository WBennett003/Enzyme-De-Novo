import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import seaborn as sns
import json 

from sklearn.metrics import confusion_matrix

class dataset(torch.utils.data.Dataset):
  def __init__(self, file_path='datasets/dense_bonding10page.h5py'):
    super().__init__()
    self.file = h5py.File(file_path, 'r+')

  def __len__(self):
    return len(self.file['reactant_elem'])
  
  def __getitem__(self, idx):
    return (self.file['reactant_elem'][idx], self.file['reactant_charge'][idx], self.file['reactant_pos'][idx],
    self.file['reactant_adj'][idx], 
    self.file['product_elem'][idx], self.file['product_charge'][idx], self.file['product_pos'][idx],
    self.file['product_adj'][idx], 
    self.file['theozyme_elem'][idx], self.file['theozyme_pos'][idx], self.file['theozyme_adj'][idx])

def plot_bonding_confusion(Y_true, Y_pred, n_bonds=8, get_figure=False, annot=False):
    bonds = np.arange(n_bonds)
    if len(Y_true.shape) == 2:
        Y_true = Y_true[None, :, :]
        Y_pred = Y_pred[None, :, :]

    conf_total = np.zeros((n_bonds, n_bonds))
    for batch in range(Y_true.shape[0]):
        for row in range(Y_true.shape[1]):
            conf_matrix = confusion_matrix(Y_true[batch][row], Y_pred[batch][row], labels=bonds)
            conf_total += conf_matrix
    
    fig, ax = plt.subplots(3, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(8,16))

    uniques = np.arange(n_bonds)
    blank = np.zeros(n_bonds)
    True_uniques, True_count = np.unique(Y_true, return_counts=True)
    blank[:True_count.shape[0]] = True_count    
    True_count = blank

    ax[1].bar(uniques, True_count)
    ax[1].set_title("True bond Distrubution")

    blank = np.zeros(n_bonds)
    Pred_uniques, Pred_count = np.unique(Y_pred, return_counts=True)
    blank[:Pred_count.shape[0]] = Pred_count    
    Pred_count = blank

    ax[2].bar(uniques, Pred_count)
    ax[2].set_title("Pred bond Distrubution")

    heatmap = sns.heatmap(conf_total, annot=annot, ax=ax[0])
    ax[0].set(xlabel='Pred', ylabel='True')

    if get_figure:
        return fig
    else:
        plt.show()
    

def plot_elem_confusion(Y_true, Y_pred, labels='PERIODIC.json', annot=False, n_elems=30, get_figure=False):
    with open(labels, 'r') as f:
        labels = json.load(f)
        labels = {v: k for k, v in labels.items()}
        labels = [labels[k] for k in sorted(labels)[:n_elems-1]]
        labels.insert(0, "Padding")

    fig, ax = plt.subplots(3, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(8,16))
    
    uniques = np.arange(n_elems)
    blank = np.zeros(n_elems)
    True_uniques, True_count = np.unique(Y_true, return_counts=True)
    blank[:True_count.shape[0]] = True_count    
    True_count = blank

    ax[1].bar(uniques, True_count)
    ax[1].set_xticks(uniques, labels)
    ax[1].set_title("True Element Distrubution")

    blank = np.zeros(n_elems)
    Pred_uniques, Pred_count = np.unique(Y_pred, return_counts=True)
    blank[:Pred_count.shape[0]] = Pred_count    
    Pred_count = blank

    ax[2].bar(uniques, Pred_count, label=labels)
    ax[2].set_title("Pred Element Distrubution")
    ax[2].set_xticks(uniques, labels)


    if len(Y_true.shape) == 1:
        Conf_matrix = confusion_matrix(Y_true, Y_pred, labels=uniques)
    else:
        confs = np.zeros((n_elems, n_elems))
        for batch in range(Y_true.shape[0]):
            Conf_matrix = confusion_matrix(Y_true[batch], Y_pred[batch], labels=uniques)
            confs += Conf_matrix
        Conf_matrix = confs

    heatmap = sns.heatmap(Conf_matrix, xticklabels=labels, yticklabels=labels, annot=annot, ax=ax[0])
    ax[0].set(xlabel='Pred', ylabel='True')

    if get_figure:
        return fig
    else:
        plt.show()

def plot_dataset(sample):
    RE, RC, RP, RADJ, PE, PC, PP, PADJ, TE, TP, TADJ = sample
    

    fig, ax = plt.subplots(3)

    sns.heatmap(RADJ, ax=ax[0])
    sns.heatmap(PADJ, ax=ax[1])
    sns.heatmap(TADJ, ax=ax[2])

    plt.show()    

def get_bonds_and_positions(F, A):
    compound_bonds = []

    for i in range(A.shape[0]):
        for j in range(A.shape[1]-i):
            if A[i][j] != 0 and sum(F[i] - F[j]) != 0:
                bond = np.array([F[i], F[j]]).T
                compound_bonds.append(bond)

    return F, compound_bonds

def get_element_colors(elements, element_colours={"O" : "red", "N" : "blue", "C" : "green", "S" : "yellow", "P" : "purple"}, element_dir='datasets/PERIODIC.json'):
    with open(element_dir, 'r') as f:
        labels = json.load(f)

    coloured_idx = {labels[k] : element_colours[k] for k in element_colours.keys()}

    colors = []
    for element in elements:
        if element in coloured_idx.keys():
            colors.append(coloured_idx[element])
        else:
            colors.append("Black")
            
    return colors

def plot_bonding_matrix(TrueADJ, PredADJ, get_figure=False):

    fig, ax = plt.subplots(1, 2, figsize=(20,8))
    sns.heatmap(TrueADJ, ax=ax[0])
    ax[0].set_title("True bonding matrix")

    sns.heatmap(PredADJ, ax=ax[1])
    ax[1].set_title("True bonding matrix")
    if get_figure:
        return fig
    else:
        plt.show()

def plot_prediction(pred, true, get_figure=False): #makes a 3d plot of the system


    TE, TP, TADJ = true
    predE, predP, predADJ = pred
    
    True_r, True_bonds = get_bonds_and_positions(TP, TADJ) #get position of atoms and bonds between them
    
    Pred_r, Pred_bonds = get_bonds_and_positions(predP, predADJ)

    True_E = get_element_colors(TE)

    Pred_E = get_element_colors(predE)

    True_r = np.array(True_r).T
    Pred_r = np.array(Pred_r).T


    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax1.scatter(True_r[0], True_r[1], True_r[2], c=True_E)
    for bond in True_bonds:
        ax1.plot(bond[0], bond[1], bond[2], color='black')

    ax2.scatter(Pred_r[0], Pred_r[1], Pred_r[2], c=Pred_E)
    for bond in Pred_bonds:
        ax2.plot(bond[0], bond[1], bond[2], color='black')

    if not get_figure:
        plt.show()
    else:
        return fig

def plot_sample(sample, colours=['red', 'blue', 'green'], get_figure=False): #makes a 3d plot of the system
    RE, RC, RP, RADJ, PE, PC, PP, PADJ, TE, TP, TADJ = sample


    
    reactants_r, reactant_bonds = get_bonds_and_positions(RP, RADJ) #get position of atoms and bonds between them
    
    products_r, product_bonds = get_bonds_and_positions(PP, PADJ)
  
    residues_r, residue_bonds = get_bonds_and_positions(TP, TADJ)

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

    if not get_figure:
        plt.show()
    else:
        return fig


if __name__ =='__main__':
    data = dataset()

    q = ''
    while q != 'Quit':
        q = input("id : ")
        plot_prediction(data[int(q)][-3:], data[int(q)][-3:])
        plot_bonding_matrix(data[int(q)][-1:][0], data[int(q)][-1:][0])
        # plot_dataset(data[int(q)])

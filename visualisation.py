import matplotlib.pyplot as plt
import numpy as np
import json 


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
    plot_mechanism_step('macie_entry_1_1_1.mrv')
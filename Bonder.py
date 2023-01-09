import json
import numpy as np

class Builder():
    def __init__(self, path='datasets/Amino Acids.json'):
        with open(path, 'r') as f:
            self.AA_bonds = json.load(f)

    def build_pdb2(self, atoms, residues):
        bond_idx = []
        bond_type = []

        idx = 0
        res = residues[idx]
        
        while idx != len(residues):

            res_atoms = self.AA_bonds[res]["Atoms"]
            length = len(res_atoms)
            sele_atoms = atoms[idx:idx+length]

            if sorted(res_atoms) != sorted(sele_atoms): #checks if atoms are the same
                return [], [] #error in building bonds so skip this protein

            bonds = self.AA_bonds[res]["Bonds"]
            for bond in bonds:
                i = sele_atoms.index(bond[0])    
                j = sele_atoms.index(bond[1])    
                bond_idx.append([i+idx,j+idx])
                bond_type.append(bond[2])
            
            idx += length
            if idx < len(residues):
                res = residues[idx]


        return bond_idx, bond_type
        
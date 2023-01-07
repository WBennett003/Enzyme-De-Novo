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
        
        while idx == len(residues):

            res_atoms = self.AA_bonds[res]["Atoms"]
            length = len(res_atoms)
            sele_atoms = atoms[idx:idx+length]
            assert res_atoms.sort() == sele_atoms.sort() #checks if atoms are the same

            bonds = self.AA_bonds[res]["Bonds"]
            for bond in bonds:
                i = sele_atoms.index(bond[0])    
                j = sele_atoms.index(bond[1])    
                bond_idx.append([i,j])
                bond_type.append(bond[2])

            idx += length

        return bond_idx, bond_type
        
    def build_pdb(self, pdb_array):
        sizes = []

        residues = []
        atom_names = []
        sorted_map = {}
        for res_id in pdb_array:
            res_type = pdb_array[res_id][1]
            residues.append(res_type)
            sorted_map[res_id] = {}
            temp = []
            sizes.append(len(pdb_array[res_id]))
            for atom in pdb_array[res_id]:
                atom_name = atom[2]
                temp.append(atom_name)
                sorted_map[res_id][atom_name] = atom
            atom_names.append(temp)

        length = sum(sizes)
        zeros = np.zeros((length, length, 2))
        count = 0
        for res_count, res in enumerate(sorted_map):
            res_type = residues[res_count][1]
            bonds = self.AA_bonds[res_type]['Bonds']
            for bond in bonds:
                try:
                    i = count + atom_names[res_count].index(bond[0])
                    j = count + atom_names[res_count].index(bond[1])
                    zeros[i, j] = [bond[2], 0]
                    zeros[j, i] = [bond[2], 0]
                except ValueError as e:
                    print(f"Atom missing on residue {res}")
    
            count += len(atom_names[res_count])
        
        self.map = sorted_map

        return zeros.tolist()
import numpy as np
import requests
import json 
import os
import h5py

from sklearn.preprocessing import StandardScaler

from utils import download_molecule, fetch_molecules, dense_bonding, combine_square_matrix, create_graph_from_mol, get_theozyme, download_protein, mask_bond_vectors
from tokeniser import Element_Tokeniser, Theozyme_Tokeniser
from Bonder import Builder

ELEMENT_TOKENISER = Element_Tokeniser('datasets/PERIODIC.json')
THEOZYME_TOKENISER = Theozyme_Tokeniser('datasets/THEOZYME.json')

class Collector():
    def __init__(self, COMPOUND_SIZE=250,
    COMPOUND_CHANNELS=5, THEOZYME_SIZE=250, THEOZYME_CHANNELS=7, N_BOND_TYPES=8,
    N_STERO_TYPES=8, CHARGE_RANGE=(-4,+4),file_dir='datasets/dense_bonding10page.h5py',
     enzyme_url='https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json'):
        self.COMPOUND_SIZE = COMPOUND_SIZE
        self.COMPOUND_CHANNELS = COMPOUND_CHANNELS
        self.THEOZYME_SIZE = THEOZYME_SIZE
        self.THEOZYME_CHANNELS = THEOZYME_CHANNELS
        self.N_BOND_TYPES = N_BOND_TYPES
        self.N_STERO_TYPES = N_STERO_TYPES
        self.CHARGE_RANGE = CHARGE_RANGE
        self.enzyme_url = enzyme_url

        self.scaler = StandardScaler()
        self.builder = Builder()
        self.file = h5py.File(file_dir, 'w')

        self.process_response(7)
        self.close_file()

    def combine_compounds(self, compounds):
        compound_elem = []
        compound_pos = []
        compound_charge = []
        compound_bond_index = []
        compound_bond_types = []
        
        n_nodes = 0
        for compound in compounds:
            elem, charge, pos, bond_idx, bond_type = create_graph_from_mol(compound, self.N_BOND_TYPES, self.N_STERO_TYPES, self.CHARGE_RANGE)
            
            global_bond_idx = []
            for idx in bond_idx:
                global_bond_idx.append([idx[0]+n_nodes, idx[1]+n_nodes])

            n_nodes += len(elem)

            compound_elem.extend(elem)
            compound_charge.extend(charge)
            compound_pos.extend(pos)
            compound_bond_index.extend(global_bond_idx)
            compound_bond_types.extend(bond_type)

        compound_elem, compound_charge, compound_pos = self.preprocess_compound(compound_elem, compound_charge, compound_pos)
        
        if len(compound_elem) == 0:
            return [], [], [], []
        else:
            compound_adj = dense_bonding(compound_bond_index, compound_bond_types, self.COMPOUND_SIZE)


        return compound_elem, compound_charge, compound_pos, compound_adj


    def get_enzymedata(self, url, i):
        url = url + '&page=' + str(i)
        return requests.get(url).json()['results']

    def close_file(self):
        self.file.close()

    def preprocess_compound(self, element, charge, position):
        max_length = self.COMPOUND_SIZE
        assert len(element) == len(charge) and len(element) == len(position)
        length = len(element)
        if length > max_length:
            return [], [], []

        blank = np.zeros(max_length, dtype='int32') #np.repeat(int('-inf'), max_length)
        blank[:length] = element
        element = blank

        blank = np.zeros(max_length, dtype='int32')
        blank[:length] = charge
        charge = blank

        blank = np.zeros((max_length, 3), dtype='float32')
        blank[:length] = self.scaler.fit_transform(np.array(position))
        position = blank
        return element, charge, position

    def process_response(self, n=5):
        self.file.create_dataset("reactant_elem", (0, self.COMPOUND_SIZE), dtype='i', chunks=True, maxshape=(None, self.COMPOUND_SIZE))
        self.file.create_dataset("reactant_charge", (0, self.COMPOUND_SIZE), dtype='i', chunks=True, maxshape=(None, self.COMPOUND_SIZE))
        self.file.create_dataset("reactant_pos", (0, self.COMPOUND_SIZE, 3), dtype='f', chunks=True, maxshape=(None, self.COMPOUND_SIZE, 3))
        self.file.create_dataset("reactant_adj", (0, self.COMPOUND_SIZE, self.COMPOUND_SIZE) , dtype='i', chunks=True, maxshape=(None, self.COMPOUND_SIZE, self.COMPOUND_SIZE))

        self.file.create_dataset("product_elem", (0, self.COMPOUND_SIZE), dtype='i', chunks=True, maxshape=(None, self.COMPOUND_SIZE))
        self.file.create_dataset("product_charge", (0, self.COMPOUND_SIZE), dtype='i', chunks=True, maxshape=(None, self.COMPOUND_SIZE))
        self.file.create_dataset("product_pos", (0, self.COMPOUND_SIZE, 3), dtype='f', chunks=True, maxshape=(None, self.COMPOUND_SIZE, 3))
        self.file.create_dataset("product_adj", (0, self.COMPOUND_SIZE, self.COMPOUND_SIZE), dtype='i', chunks=True, maxshape=(None,self.COMPOUND_SIZE, self.COMPOUND_SIZE))


        self.file.create_dataset("theozyme_elem", (0, self.THEOZYME_SIZE), dtype='i', chunks=True, maxshape=(None,self.THEOZYME_SIZE))
        self.file.create_dataset("theozyme_pos", (0, self.THEOZYME_SIZE, 3), dtype='f', chunks=True, maxshape=(None,self.THEOZYME_SIZE, 3))
        self.file.create_dataset("theozyme_adj", (0, self.THEOZYME_SIZE, self.THEOZYME_SIZE), dtype='i', chunks=True, maxshape=(None,self.THEOZYME_SIZE,self.THEOZYME_SIZE))


        for i in range(1, n):
            print(f"Getting page {i}!")
            self.enzyme_response = self.get_enzymedata(self.enzyme_url, i)
            for result in self.enzyme_response:
                # process compounds
                compounds = result['reaction']['compounds']
                reactants, products = fetch_molecules(compounds)

                if len(products) == 0:
                    print(f"Error, no products found {compounds}")
                    break

                if len(reactants) == 0:
                    print(f"Error, no reactants found {compounds}")
                    break

                reactant_elem, reactant_charge, reactant_pos, reactant_adj = self.combine_compounds(reactants)
                

                    
                product_elem, product_charge, product_pos, product_adj = self.combine_compounds(products)

                if len(product_elem) == 0:
                    print(f"Error, products failed {len(products)}")
                    break

                if len(reactant_elem) == 0:
                    print(f"Error, reactants failed {len(reactants)}")
                    break

                proteins = get_theozyme(result)

                for protein in proteins:
                    elem, pos, adj = self.generate_theozyme(protein, proteins[protein])

                    if isinstance(elem, type(np.array([]))):
                        self.save_dataset('reactant_elem', reactant_elem)
                        self.save_dataset('reactant_charge', reactant_charge)
                        self.save_dataset('reactant_pos', reactant_pos)
                        self.save_dataset('reactant_adj', reactant_adj)

                        self.save_dataset('product_elem', product_elem)
                        self.save_dataset('product_charge', product_charge)
                        self.save_dataset('product_pos', product_pos)
                        self.save_dataset('product_adj', product_adj)

                        self.save_dataset('theozyme_elem', elem)
                        self.save_dataset('theozyme_pos', pos)
                        self.save_dataset('theozyme_adj', adj)
                    else:
                        print(f"protein failed {[protein]}")

        print(f"dataset shapes : \n {[[x, self.file[x].shape] for x in self.file.keys()]}")


    def save_dataset(self, key, arr):
        self.file[key].resize((self.file[key].shape[0] + 1), axis=0)
        self.file[key][-1:] = arr
                       

    def generate_theozyme(self, pdb_id, ress, threshold=0):   
        pdb_data = download_protein(pdb_id)
        ress = sort(ress)
        if pdb_data == '<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">\n<html>\n  <head>\n    <title>404 Not Found</title>\n  </head>\n  <body>\n    <h1>Not Found</h1>\n    <p>The requested URL was not found on this server.</p>\n    <hr>\n    <address>RCSB PDB</address>\n  </body>\n</html>\n':
            return 0, 0, 0
        atoms = pdb_data.split('\n')
        mutations = []
        
        elem = []
        res = []
        charge = []
        pos = []

        idx = 0
        next_res = ress[idx]
        last_i = -1
        last_rand_i = 0
        wrong_res = []
        for atom in atoms:
            if atom[:4] == 'ATOM':
                a = atom[13:16].replace(' ', '')
                r = atom[17:20].capitalize()
                i = int(atom[22:26])
                x = float(atom[30:38])
                y = float(atom[38:46])
                z = float(atom[46:55])
                t = float(atom[60:66])
                
                if i == last_i+1 and idx < len(ress)-1 and last_rand_i != i:
                    idx += 1
                    next_res = ress[idx]

                if i == next_res[0]:
                    last_i = i
                    if r == next_res[1]:
                        elem.append(a)
                        res.append(r)
                        pos.append([x,y,z])
                    elif next_res not in wrong_res:
                        wrong_res.append(next_res)
                        print(f"wrong residue at position {i} got {r} instead of {next_res[1]}")
                        mutations.append(next_res)

                last_rand_i = i


        if len(mutations) <= threshold and len(res) != 0:
            bond_idx, bond_types = self.builder.build_pdb2(elem, res)
            if len(bond_idx) == 0:
                return 0, 0, 0
                
            tokenised_elem = []
            
            for i in range(len(elem)):
                tokenised_elem.append(ELEMENT_TOKENISER.tokenise(elem[i]))
            
            n_atoms = len(elem)
            if n_atoms > self.THEOZYME_SIZE:
                return 0, 0, 0

            zeros = np.zeros(self.THEOZYME_SIZE)#np.repeat(int('-inf'), self.THEOZYME_SIZE)
            zeros[:len(tokenised_elem)] = tokenised_elem
            elem = np.array(zeros)

            zeros = np.zeros((self.THEOZYME_SIZE, 3))
            zeros[:len(pos)] = self.scaler.fit_transform(np.array(pos))
            pos = zeros

            adj = dense_bonding(bond_idx, bond_types, self.THEOZYME_SIZE)

            return elem, pos, adj
        
        return 0, 0, 0 # cringe

def last(n):
    return n[0]

def sort(tuples):
    return sorted(tuples, key=last)

if __name__ == '__main__':
    collector = Collector()
    
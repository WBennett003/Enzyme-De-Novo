import numpy as np
import requests
import json 
import os

from rdkit import Chem
from rdkit.Chem import AllChem

from tokeniser import Element_Tokeniser

ELEMENT_TOKENISER = Element_Tokeniser()

def download_molecule(chebi_id, path='molecules/'):
    if not os.path.isfile(path+chebi_id+'.mol'):
        url = f'http://www.ebi.ac.uk/thornton-srv/m-csa/media/compound_mols/{chebi_id}.mol'
        data = requests.get(url).text

        mol = Chem.MolFromMolBlock(data)
        if mol is None:
            return True

        try: #TODO: need to make this more robust
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            data = Chem.MolToMolBlock(mol)
        except RuntimeError as e:
            print(f"compound {chebi_id} gave error {e} and shall be 2D")

        with open(path+'/'+chebi_id+'.mol', 'w+') as file:
            file.write(data)
    return False

def mask_bond_vectors(bond_idx, bond_types, max_length=500):
    n = len(bond_types)

    #create a masked_array of length 2 x 'max_length' for bond indexes 
    bond_idx = np.array(bond_idx).T
    blank = np.zeros((2, max_length))
    blank[:, :n] = bond_idx
    bond_idx = blank
    mask = np.ones((2, max_length))
    entries = np.zeros((2,n))
    mask[:, :n] = entries
    bond_idx_ma = np.ma.MaskedArray(data=bond_idx, mask=mask)

    #create a masked_array of length 'max_length' for bond types
    blank = np.zeros(max_length)
    blank[:n] = bond_types
    bond_types = blank
    mask = np.ones(max_length)
    entries = np.zeros(n)
    mask[:n] = entries
    bond_type_ma = np.ma.MaskedArray(data=bond_types, mask=mask)

    return bond_idx_ma, bond_type_ma



def download_protein(pdb_code, path='proteins/'):
    if not os.path.isfile(path+pdb_code+'.pdb'):
        pdb = (requests.get('https://files.rcsb.org/download/'+pdb_code+'.pdb').content).decode('UTF-8')
        with open(path+pdb_code+'.pdb', 'w+') as f:
            f.write(pdb)
    else:
        with open(path+pdb_code+'.pdb', 'r') as f:
            pdb = f.read()
    return pdb

def get_theozyme(result): #returns 2 tuple of (RES, ID)
    residue = result['residues']

    if len(residue) == 0:
        print(f"Error, sample has missing residues :\n {residue}")

    pdb = {}
    for res in residue:
        for i, r in enumerate(res['residue_sequences']):
            if len(res['residue_chains']) > 0:
                pdb_id = res['residue_chains'][i]['pdb_id']
                res_id = ( res['residue_chains'][i]['auth_resid'],  res['residue_chains'][i]['code'])
                if pdb_id not in pdb.keys():
                    pdb[pdb_id] = []
                pdb[pdb_id].append(res_id)

    if len(pdb) == 0:
        print(f"Error, sample has missing pdb or residue atoms :\n pdb : {pdb} ")
    
    return pdb

def fetch_molecules(compound):
    reactants = []
    products = []
    for c in compound:
        count = c['count']
        t = c['type']
        chem_id = c['chebi_id']
        error = download_molecule(chem_id) #download the molecule structure
        if error:
            return [], []
        if t == 'reactant':
            for i in range(count):
                reactants.append(chem_id)
        else:
            for i in range(count):
                products.append(chem_id)
    return reactants, products

def fetch_molecule_smiles(compounds):
    reactants = []
    products = []
    for c in compounds:
        count = c['count']
        t = c['type']
        chem_id = c['chebi_id']
        error = download_molecule(chem_id) #download the molecule structure
        if error:
            return 0, 0
        if t == 'reactant':
            for i in range(count):
                reactants.append(chem_id)
        else:
            for i in range(count):
                products.append(chem_id)
    return reactants, products


def create_graph_from_mol(chebi_id, bond_types, stero_types, charge_range, path='molecules'):
    
    charge_modifier = charge_range[1] - charge_range[0]

    with open(path+'/'+chebi_id+'.mol', 'r') as file:
        lines = file.readlines()

    n_atoms = int(lines[3][:3])
    n_bonds = int(lines[3][3:6])

    elem = []
    pos = []

    bond_index = []
    bond_types = []
    
    for i in range(4, 4+n_atoms): #TODO: need to add mass to feature array instead of element embedding
        line = lines[i]
        x = float(line[:10])
        y = float(line[10:20])
        z = float(line[20:30])
        E = line[30:33].replace(' ', '')
        E = ELEMENT_TOKENISER.tokenise(E)
        C = 0
        elem.append(E)
        pos.append([x,y,z])


    for n in range(4+n_atoms, 4+n_atoms+n_bonds):
        line = lines[n]
        i = int(line[:3])-1 #first atom
        j = int(line[3:6])-1 # second atom
        t = int(line[6:9]) # type of bond, single doubel tripple or aromatic
        s = int(line[9:12]) #sterisomerism of bond

        bond_index.append([i, j])
        bond_types.append(t)

        # since its pairwise its synmetrical and so Aij = Aji
        
    count = 4+n_atoms+n_bonds+1
    done = False
    charge = np.zeros(len(elem), dtype=np.int32)

    while not done and count != len(lines): #adds chg by the ending
        line = lines[count]
        if line[3:6] == 'END':
            done = True
        elif line[3:6] == 'CHG':
            i = int(line[9:13]) - 1
            c = int(line[13:17])
            charge[i] = c + charge_modifier
        count += 1

    return elem, charge, pos, bond_index, bond_types

def convert_coo_aj(coo, data, max_length): #takes indicies for sparse tensor
    
    if coo.shape[0] != 2:
        zeros = np.zeros((coo.shape[0], max_length, max_length))
    else:
        zeros = np.zeros((max_length, max_length))
    
    zeros[coo] = data
    return zeros

def combine_square_matrix(As, max_length):
    sizes = []
    for A in As:
        sizes.append(len(A))
    length = sum(sizes)
    blank = np.zeros((max_length, max_length), dtype=np.int32)

    count = 0
    for i, A in enumerate(As):
        l = sizes[i] + count
        blank[count:l, count:l] = A
        count = l
    
    return blank
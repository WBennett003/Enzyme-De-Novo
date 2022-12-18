import numpy as np
import requests
import json 
import os


from rdkit import Chem
from rdkit.Chem import AllChem

from tokeniser import Element_Tokeniser, Theozyme_Tokeniser

ELEMENT_TOKENISER = Element_Tokeniser('datasets/PERIODIC.json')
THEOZYME_TOKENISER = Theozyme_Tokeniser('datasets/THEOZYME.json')

def download_molecule(chebi_id, path='molecules/'):
    if not os.path.isfile(path+chebi_id+'.mol'):
        url = f'http://www.ebi.ac.uk/thornton-srv/m-csa/media/compound_mols/{chebi_id}.mol'
        data = requests.get(url).text

        mol = Chem.MolFromMolBlock(data)
        mol = Chem.AddHs(mol)
        try: #TODO: need to make this more robust
            AllChem.EmbedMolecule(mol)
        except RuntimeError as e:
            data = data
        data = Chem.MolToMolBlock(mol)

        with open(path+'/'+chebi_id+'.mol', 'w+') as file:
            file.write(data)

def download_protein(pdb_code, path='proteins/'):
    if not os.path.isfile(path+pdb_code+'.pdb'):
        pdb = (requests.get('https://files.rcsb.org/download/'+pdb_code+'.pdb').content).decode('UTF-8')
        with open(path+pdb_code+'.pdb', 'w+') as f:
            f.write(pdb)
    else:
        with open(path+pdb_code+'.pdb', 'r') as f:
            pdb = f.read()
    return pdb


def create_matrix_from_mol(chebi_id, path='molecules'):

    with open(path+'/'+chebi_id+'.mol', 'r') as file:
        lines = file.readlines()

    n_atoms = int(lines[3][:3])
    n_bonds = int(lines[3][3:6])

    F = []
    A = np.zeros((n_atoms, n_atoms, 2))

    for i in range(4, 4+n_atoms): #TODO: need to add mass to feature array instead of element embedding
        line = lines[i]
        x = float(line[:10])
        y = float(line[10:20])
        z = float(line[20:30])
        E = line[30:33].replace(' ', '')
        E = ELEMENT_TOKENISER.tokenise(E)
        C = 0
        F.append([E, x, y, z, C])

    for n in range(4+n_atoms, 4+n_atoms+n_bonds):
        line = lines[n]
        i = int(line[:3])-1 #first atom
        j = int(line[3:6])-1 # second atom
        t = int(line[6:9]) # type of bond, single doubel tripple or aromatic
        s = int(line[9:12]) #sterisomerism of bond

        # since its pairwise its synmetrical and so Aij = Aji
        A[i,j][0] = t
        A[i,j][1] = s
        A[j,i][0] = t
        A[j,i][1] = s

    count = 4+n_atoms+n_bonds+1
    done = False
    while not done and count != len(lines): #adds chg by the ending
        line = lines[count]
        if line[3:6] == 'END':
            done = True
        elif line[3:6] == 'CHG':
            i = int(line[9:13]) - 1
            c = int(line[13:17])
            F[i][4] = c
        count += 1

    return F, A.tolist()

def create_system_matrix(Fs, As):
    #Fs is a list of Features of a molecule, As are ajacency Matrices of a molecule
    n = len(Fs)
    Master_F = []
    A_sizes = []
    for i in range(n):
        F, A = Fs[i], As[i]
        
        Master_F.extend(F) # TODO: need to make cooridinates from two molecules not overlap or interact
        A_sizes.append(A.shape[0])
    
    size = sum(A_sizes)
    Master_A = np.zeros((size, size, 2))

    last_step = 0
    for i,A in enumerate(As):
        step = last_step+A_sizes[i]
        Master_A[last_step:step, last_step:step] = A
        last_step = step
    
    return Master_F, Master_A

def create_system(compounds):
    Fs = []
    As = []
    for compound in compounds:
        F, A = create_matrix_from_mol(compound)
        Fs.append(F), As.append(A)

    # system_F, system_A = create_system_matrix(Fs, As)   
    return Fs, As

def preprocess_residue(theo):
    temp = []
    for residues in theo:
        for res in residues:
            for atom in residues[res]:
                i = atom[0]
                AA = atom[1]
                AA = THEOZYME_TOKENISER.tokenise(AA) #converts string to index for one hot embedding
                elem = atom[2]
                elem = ELEMENT_TOKENISER.tokenise(elem)
                x, y, z, t = atom[3], atom[4], atom[5], atom[6] #TODO: add position normalisation
                temp.append([i, AA, elem, x, y, z, t])
    return temp

class Collector():
    def __init__(self, enzyme_url='https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json'):
        self.enzyme_response = self.get_enzymedata(enzyme_url)
        self.raw_enzyme_data = self.extract_features() #contains the mol and pdb files involved with the enzymes 
        self.enzyme_data_index = self.process_enzyme_data()
        self.enzyme_dataset = self.generate_dataset()

    def get_enzymedata(self, url):
        return requests.get(url).json()['results']

    def get_reaction(self, result): #returns reactants and products
        compounds = result['reaction']['compounds']
        return compounds

    def get_theozyme(self, result): #returns 2 tuple of (RES, ID)
        residue = result['residues']

        if len(residue) == 0:
            print(f"Error, sample has missing residues :\n {residue}")

        temp = []
        pdb = set()
        for res in residue:
            for i, r in enumerate(res['residue_sequences']):
                temp.append((r['resid'], r['code']))
                pdb.add(res['residue_chains'][i]['pdb_id'])

        if len(pdb) == 0 or len(temp) == 0:
            print(f"Error, sample has missing pdb or residue atoms :\n pdb : {pdb} \n res atoms : {temp}")
        
        return [temp, list(pdb)]

    def extract_features(self):
        temp = {}
        for result in self.enzyme_response:
            temp[result['mcsa_id']] = {}
            temp[result['mcsa_id']]['compounds'] = self.get_reaction(result)
            temp[result['mcsa_id']]['residues'] = self.get_theozyme(result)
        return temp

    def process_compounds(self, sample):
        compound = sample['compounds']
        
        temp = {}
        temp['reactants'] = []
        temp['products'] = []
        temp['residues'] = []

        temp = self.fetch_molecules(compound, temp) #iterates over compounds

        #Checks if reactants or products are missing
        if len(temp['reactants']) == 0:
            print(f"Error, no reactants found : \n {sample}")
            return {"Error" : "No reactants"}

        if len(temp['products']) == 0:
            print(f"Error, no products found : \n {sample}")
            return {"Error" : "No products"}



        residues = sample['residues']
        ress = residues[0]
        ress_dict = dict(ress)
        INDEXES = set([i[0] for i in ress])
        RESIDUES = set([i[1] for i in ress])


        pdb = residues[1]
        if len(pdb) > 1:
            pdb = pdb[0]
        else:
            pdb = pdb[0]
        
        pdb_data = download_protein(pdb)
        atoms = pdb_data.split('\n')
        found_residues = set()
        mutations = []
        res_pos = {}
        for atom in atoms:
            if atom[:4] == 'ATOM':
                a = atom[13:16]
                r = atom[17:20].capitalize()
                i = int(atom[22:26])
                x = float(atom[30:38])
                y = float(atom[38:46])
                z = float(atom[46:55])
                t = float(atom[60:66])
                if i in INDEXES:
                    if r in RESIDUES:
                        found_residues.add(i)
                        res_id = r + str(i)
                        if res_id not in res_pos.keys():
                            res_pos[res_id] = []
                        res_pos[res_id].append([i, r, a, x, y, z, t])
                    else:
                        mutations.append([i, ress_dict[i], r])
                        # print(f"Wrong residue at position {i} should be {ress_dict[i]} instead {r}")
        
        if (found_residues != INDEXES and len(mutations) > 0) or len(res_pos) == 0:
            print(f"Error, pdb {pdb} appears to have missing residues or incorrect residues \n {mutations}")
            return {"Error" : ["pdb does not contain active site atoms", mutations]}


        temp['residues'].append(res_pos)
        
        return temp

    def fetch_molecules(self, compound, temp):
        for c in compound:
            count = c['count']
            t = c['type']
            chem_id = c['chebi_id']
            download_molecule(chem_id) #download the molecule structure
            if t == 'reactant':
                for i in range(count):
                    temp['reactants'].append(chem_id)
            else:
                for i in range(count):
                    temp['products'].append(chem_id)
        return temp

    def process_enzyme_data(self):
        skipped = 0
        new = {}
        for i in self.raw_enzyme_data:
            sample = self.raw_enzyme_data[i]
            out = self.process_compounds(sample)
            if "Error" not in out.keys():
                 new[i] = out
            else:
                skipped += 1
                print(f"skipping {i}")
        print(f"skipped {skipped}/{len(self.raw_enzyme_data)}")
        return new
    
    def generate_dataset(self, filename='datasets/X_processed_dataset.json'):
        d = self.enzyme_data_index
        dataset = {}


        for r in d:
            reactants = d[r]['reactants'] #TODO: add coordiante normalisation
            F, A = create_system(reactants)
            RF = F
            RA = A

            products = d[r]['products'] 
            F, A = create_system(products)
            PF = F 
            PA = A

            res = preprocess_residue(d[r]['residues']) 

            if len(RF) != 0 and len(PF) != 0 and len(RA) != 0 and len(PA) != 0 and len(res) != 0:
                dataset[r] = {
                'reactants' : {},
                'products'  : {},
                }
                dataset[r]['reactants']['F'] = RF
                dataset[r]['reactants']['A'] = RA
                dataset[r]['products']['F'] = PF
                dataset[r]['products']['A'] = PA
                dataset[r]['residues'] = res
            else:
                print(f"sample {r} has missing products, reactants or active site")
                
        with open(filename, 'w+') as file:
            json.dump({'X' : dataset}, file)
        return dataset

if __name__ == '__main__':
    collector = Collector()
    
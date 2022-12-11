import json
import numpy as np
from tokeniser import Element_Tokeniser, Theozyme_Tokeniser

ELEMENT_TOKENISER = Element_Tokeniser('datasets/PERIODIC.json')
THEOZYME_TOKENISER = Theozyme_Tokeniser('datasets/THEOZYME.json')

def get_reactions(path='datasets/X_dataset.json'):
    with open(path, 'r') as file:
        d = json.load(file)
    return d

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



    return F, A

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

    system_F, system_A = create_system_matrix(Fs, As)   
    return system_F, system_A

def preprocess_residue(residues):
    temp = []
    for res in residues:
        i = res[0]
        AA = res[1]
        AA = Theozyme_Tokeniser.tokenise(AA) #converts string to index for one hot embedding
        atom = res[2]
        atom = Element_Tokeniser.tokenise(atom)
        x, y, z, t = res[3], res[4], res[5], res[6] #TODO: add position normalisation
        
def preprocess_compound(compound):
    return compound
    
def generate_dataset(filename='datasets/X_processed_dataset.json'):
    d = get_reactions()
    dataset = {}


    for r in d:
        dataset[r] = {
            'reactants' : {},
            'products'  : {},
            }
        reactants = d[r]['reactants'] #TODO: add coordiante normalisation
        F, A = create_system(reactants)
        dataset[r]['reactants']['F'] = F
        dataset[r]['reactants']['A'] = A.tolist()

        products = d[r]['products'] 
        F, A = create_system(products)
        dataset[r]['products']['F'] = F 
        dataset[r]['products']['A'] = A.tolist()

        dataset[r]['residues'] = preprocess_residue(d[r]['residues']) 

    with open(filename, 'w+') as file:
        json.dump({'X' : dataset}, file)
    return dataset

if __name__ == '__main__':
    generate_dataset()

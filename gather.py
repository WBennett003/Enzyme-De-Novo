import requests
import json 
import os

def download_molecule(chebi_id, path='molecules/'):
    if not os.path.isfile(path+chebi_id+'.mol'):
        url = f'http://www.ebi.ac.uk/thornton-srv/m-csa/media/compound_mols/{chebi_id}.mol'
        data = requests.get(url).text
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
 
def process_compounds(sample):
    compound = sample['compounds']
    temp = {}
    temp['reactants'] = []
    temp['products'] = []
    temp['residues'] = []
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

    residues = sample['residues']
    ress = residues[0]
    pdb = residues[1]
    if len(pdb) == 1:
        pdb = pdb[0]
    else:
        pdb = pdb[0]

    pdb_data = download_protein(pdb)
    atoms = pdb_data.split('\n')
    res_pos = []
    for atom in atoms:
        if atom[:4] == 'ATOM':
            a = atom[13:16]
            r = atom[17:20].capitalize()
            i = int(atom[22:26])
            x = float(atom[30:38])
            y = float(atom[38:46])
            z = float(atom[46:55])
            t = float(atom[60:66])
            if [i, r] in ress:
                res_pos.append([i, r, a, x, y, z, t])
    
    temp['residues'].append(res_pos)
    return temp

def process_all_compounds(source_filename='enzyme_dataset.json', output_filename='X_dataset.json', path='datasets/'):
    with open(path+source_filename, 'r') as file:
        d = json.load(file)
        new = {}
        for i in d:
            sample = d[i]
            new[i] = process_compounds(sample)
                
    with open(path+output_filename, 'w+') as file:
        json.dump(new, file) 

def process_enzymes(sample):
    uniprot = sample['protein']['sequences']

def get_enzymedata(url='https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json'):
    return requests.get(url).json()['results']

def get_reaction(result): #returns reactants and products
    compounds = result['reaction']['compounds']
    return compounds

def get_mechanism(result): #returns description of mechanisms
    mechs = result['reaction']['mechanisms']
    m = []
    for mech in mechs:
        _ = []
        for step in mech['steps']:
            _.append(step['description'])
        m.append(_)
    return m

def get_theozyme(result): # returns 2 tuple of (RES, ID)
    residue = result['residues']
    temp = []
    pdb = set()
    for res in residue:
        _ = []
        for i, r in enumerate(res['residue_sequences']):
            temp.append((r['resid'], r['code']))
            pdb.add(res['residue_chains'][i]['pdb_id'])
        # temp.append(_)
    return [temp, list(pdb)]

def get_residues(result):
    residues = [i['residue_sequences'] for i in result['residues']]
    return residues

def get_uniprot(result):
    pdb = result['reference_uniprot_id']

def clean_d(results):
    temp = {}
    for result in results:
        temp[result['mcsa_id']] = {}
        temp[result['mcsa_id']]['compounds'] = get_reaction(result)
        temp[result['mcsa_id']]['residues'] = get_theozyme(result)
        temp[result['mcsa_id']]['uniprot'] = get_uniprot(result)
        # temp[result['mcsa_id']]['mechanisms'] = get_mechanism(result)
    return temp

def make_dataset():
    d = get_enzymedata()
    d = clean_d(d)
    with open('datasets/enzyme_dataset.json', 'w+') as f:
        json.dump(d, f)
    
if __name__ == '__main__':
    make_dataset()
    process_all_compounds()

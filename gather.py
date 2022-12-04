import requests
import json 
import os

def download_molecule(chebi_id, path='molecules'):
    if not os.path.isfile(path+'/'+chebi_id+'.mol'):
        url = f'http://www.ebi.ac.uk/thornton-srv/m-csa/media/compound_mols/{chebi_id}.mol'
        data = requests.get(url).text
        with open(path+'/'+chebi_id+'.mol', 'w+') as file:
            file.write(data)

def download_protein():
    pass


def process_compounds(sample):
    compound = sample['compounds']
    temp = {}
    temp['reactants'] = []
    temp['products'] = []
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

def process_all_compounds(filename='enzyme_dataset.json'):
    with open(filename, 'r') as file:
        d = json.load(file)
        new = {}
        for i in d:
            sample = d[i]
            new[i] = process_compounds(sample)
                
    with open('X_dataset.json', 'w+') as file:
        json.dump(new, file) 

def process_enzymes(sample):
    uniprot = sample['uniprot_id']

def get_enzymedata(url='https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json'):
    return requests.get(url).json()['results']

def get_reaction(result): #returns
    compounds = result['reaction']['compounds']
    return compounds

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
        temp[result['mcsa_id']]['residues'] = get_residues(result)
        temp[result['mcsa_id']]['uniprot'] = get_uniprot(result)
    return temp

if __name__ == '__main__':
    process_all_compounds()

import json

def process_compounds(sample):
    compound = sample['compounds']
    reactants = []
    products = []
    for c in compound:
        count = c['count']
        t = c['type']
        chem_id = c['chebi_id']

with open('enzyme_data.csv', 'r') as file:
    d = json.load(file)


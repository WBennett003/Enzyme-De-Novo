import json
import os


class Element_Tokeniser():
    def __init__(self, dict_path='PERIODIC.json'):
        self.dict_path = dict_path
        if not os.path.isfile(self.dict_path):
            self.PERIODIC_TABLE = {}
        else:
            with open(self.dict_path, 'r') as f:
                self.PERIODIC_TABLE = json.load(f)

        self.elements = self.PERIODIC_TABLE.keys()

    def tokenise(self, element):
        element = element.replace(' ', '')
        if element.capitalize() not in self.elements:
            n = len(self.elements)
            self.PERIODIC_TABLE[element] = n
            self.update_table()
        else:
            n = self.PERIODIC_TABLE[element]
        return n    

    def update_table(self):
        with open(self.dict_path, 'w+') as f:
            json.dump(self.PERIODIC_TABLE, f) 

class Theozyme_Tokeniser():
    def __init__(self, dict_path='Theozyme.json'):
        self.dict_path = dict_path
        if not os.path.isfile(self.dict_path):
            self.THEOZYME_TABLE = {}
        else:
            with open(self.dict_path, 'r') as f:
                self.THEOZYME_TABLE = json.load(f)

        self.components = self.THEOZYME_TABLE.keys()

    def tokenise(self, component):
        component = component.replace(' ', '')
        if component.capitalize() not in self.components:
            n = len(self.components)
            self.THEOZYME_TABLE[component] = n
            self.update_table()
        else:
            n = self.THEOZYME_TABLE[component]
        return n    

    def update_table(self):
        with open(self.dict_path, 'w+') as f:
            json.dump(self.THEOZYME_TABLE, f) 
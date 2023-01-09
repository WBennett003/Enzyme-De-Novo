import json
import os


class Element_Tokeniser():
    position_chars = [
        'A', 'B', 'G', 'D', 'E', 'Z'
    ]


    def __init__(self, dict_path='PERIODIC.json'):
        self.dict_path = dict_path
        if not os.path.isfile(self.dict_path):
            self.PERIODIC_TABLE = {}
        else:
            with open(self.dict_path, 'r') as f:
                self.PERIODIC_TABLE = json.load(f)

    def clean_element_name(self, element):
        element = element.replace(' ', '')
        if len(element) > 1:
            if element[1] in self.position_chars:
                return element[0]
            elif element[1].isdigit():
                return element[0]
            elif element[1].lower() != element[1]:
                return element[0]
            elif element[1].lower() == element[1] and len(element) == 3:
                return element[:-1]
            else:
                return element
        else:
            return element

    def tokenise(self, element):
        element = self.clean_element_name(element)
        if element not in self.PERIODIC_TABLE.keys():
            n = len(self.PERIODIC_TABLE.keys())
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
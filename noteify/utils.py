# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:44:34 2020

@author: Shahir
"""

import os
import abc
import json

import noteify

class JSONDictSerializable(metaclass=abc.ABCMeta):
    def __str__(self):
        return str(self.to_dict())
    
    def __repr__(self):
        return str(self.to_dict())
    
    @abc.abstractmethod
    def to_dict(self):
        pass
    
    @abc.abstractclassmethod
    def from_dict(cls, dct):
        pass

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data):
        return cls.from_dict(json.loads(data))

    def to_bytes(self):
        return self.to_json().encode()

    @classmethod
    def from_bytes(cls, data):
        return cls.from_json(data.decode())

def number_menu(option_list):
    print("-"*60)
    
    for n in range(len(option_list)):
        print(n, ": " , option_list[n])
    
    choice = input("Choose the number corresponding to your choice: ")
    for n in range(5):        
        try: 
            choice = int(choice)
            if choice < 0 or choice > len(option_list)-1:
                raise ValueError    
            print("-"*60 + "\n")
            return choice, option_list[choice]
        except ValueError: 
            choice = input("Invalid input, choose again: ")
    
    raise ValueError("Not recieving a valid input")

def get_rel_pkg_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(noteify.__file__), "..", path))
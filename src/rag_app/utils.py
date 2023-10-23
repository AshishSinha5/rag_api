import json
import pickle
import yaml

def read_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    return data

def load_yaml_file(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data
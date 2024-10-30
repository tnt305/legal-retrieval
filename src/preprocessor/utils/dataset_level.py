import pickle
import json
import os

def read_json(folder : str):
    with open(folder, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def save_json(data_list: list, folder: str):
    #'./data/update_vocab.json'
    with open(folder, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

def read_pickle(folder: str, data):
    with open(folder, 'wb') as f:
        pickle.dump(data, f)

def save_pickle(folder: str):
    with open(folder, 'rb') as f:
        data = pickle.load(f)
    return data
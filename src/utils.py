import json
import pickle

def open_json(path):
    
    with open(path, "rb") as file:
        ret_dict = json.load(file)

    return ret_dict

def dict_to_json(dict_to_save, path):
    with open(path, 'w') as file:
        json.dump(dict_to_save, file)


def open_pickle(path):
    with open(path, 'rb') as file:
        responses = pickle.load(file)
    return responses


def save_pickle(content, path):

    with open(path, 'wb') as file:
        pickle.dump(content, file)

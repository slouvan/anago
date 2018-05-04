import io
import os
import zipfile

import requests
import yaml
from os import listdir
from os.path import isfile, join
import os
import glob







def download(url, save_dir='.'):
    """Downloads trained weights, config and preprocessor.

    Args:
        url (str): target url.
        save_dir (str): store directory.
    """
    print('Downloading...')
    r = requests.get(url, stream=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as f:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        f.extractall(save_dir)
    print('Complete!')


def load_config_file(filename):
    with open(filename, "r") as f:
        cfg = yaml.load(f)
    return cfg


def get_best_model_file(dir_name):
    print("Hello")
    files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    max_f1_score = -1
    best_model_file_name = ""
    for file in files:
        if file.endswith(".h5") and len(file.split("_")) == 4:
            #print(file)
            fields = file.split("_")
            #print(fields)
            #print(fields[1])
            curr_f1 = float(fields[3].replace(".h5",""))
            #print(curr_f1)
            if curr_f1 > max_f1_score:
                max_f1_score = curr_f1
                best_model_file_name = file

    print("Loading the best model from {} ".format(os.path.join(dir_name,best_model_file_name)))
    return best_model_file_name

def clean_dir(dir_name):
    files = glob.glob(dir_name+"/*")
    for f in files:
        os.remove(f)
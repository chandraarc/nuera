import pandas as pd
import json


def load_json_data(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data





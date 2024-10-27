import pandas as pd

def read_csv(file_name):
    return pd.read_csv(file_name)

def read_json_data_frame(file_name):
    return pd.read_json(file_name)
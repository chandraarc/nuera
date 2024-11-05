import pandas as pd
import os

def read_csv(file_name):
    if not os.path.exists(file_name):
      return pd.DataFrame()
    return pd.read_csv(file_name)

def read_json_data_frame(file_name):
    return pd.read_json(file_name)
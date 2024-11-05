import numpy as np
from sklearn.impute import KNNImputer
import pandas as pd
def replace_empty(data_frame):
    imputer = KNNImputer(n_neighbors=5)
    columns = data_frame.columns
    data_frame = imputer.fit_transform(data_frame)
    data_frame = pd.DataFrame(data_frame, columns=columns)
    print(data_frame)
    return data_frame


def get_unique_values(group):
    dataMap = {}
    for index, value in enumerate(group):
        dataMap[value] = index
    return dataMap

def convert_value_key(map):
    result = {y: x for x, y in map.items()}
    return result

def is_digit_column(columnValue):
    if str(columnValue)=='nan':
        return False
    try:
        float(columnValue)
        return True
    except ValueError:
        return False


def get_categorical_labels(df):
    categorical_labels = []
    for column in df.columns:
        if is_digit_column(df[column][0]):
            columnValue = df[column][0]
        elif (df[column][0] == None or str(df[column][0]) == 'nan' or df[column][0] == 'NaN' or df[column][0] == 'NA'):
            columnValue = df[column][0]
            isNotDigit = False
            for i in range(len(df)):
                columnValue = df[column][i]
                if (str(columnValue) != 'nan' and not is_digit_column(columnValue)):
                    isNotDigit = True
                    break
            if isNotDigit:
                categorical_labels.append(column)
        else:
            categorical_labels.append(column)
    return categorical_labels


def process(instanceName, data_frame, categorical_labels_list, ignore_labels):
    data_after_process = {}
    categoricals = {}
    for label in ignore_labels:
        data_frame = data_frame.drop([label], axis=1)
    categorical_labels = get_categorical_labels(data_frame)
    categorical_labels.extend(categorical_labels_list)
    categorical_labels = list(set(categorical_labels))
    for label in categorical_labels:
        print(instanceName)
        print(label)
        result = get_unique_values(list(set(data_frame[label])))
        print(result)
        categoricals[label] = result
        data_frame[label] = data_frame[label].map(result)

    data_frame=replace_empty(data_frame)
    data_after_process['result'] = data_frame
    data_after_process['map'] = categoricals
    return data_after_process


def map_labels(data_frame, map):
    categorical_labels = get_categorical_labels(data_frame)
    for label in categorical_labels:
        result = map[label]
        data_frame[label] = data_frame[label].map(result)
    return data_frame


def drop_labels(data_frame, ignore_labels):
    data_frame =data_frame.drop(ignore_labels, axis=1)
    return data_frame


def convert_to_float(df):
    for col in df.columns:
        print(df[col])
        df[col] = df[col].astype(float)

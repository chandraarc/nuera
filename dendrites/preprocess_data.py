import numpy as np

def replace_empty(data_frame):
    data_frame = data_frame.replace('', np.nan)
    data_frame.fillna(0, inplace=True)

def get_unique_values(group):
    dataMap ={}
    for index, value in enumerate(group):
         dataMap[value] = index
    return dataMap


def process(instanceName, data_frame, categorical_labels, ignore_labels):
    data_after_process = {}
    categoricals = {}
    for label in ignore_labels:
        data_frame = data_frame.drop(label, axis=1)

    for label in categorical_labels:
        print(instanceName)
        print(label)
        result = get_unique_values(list(set(data_frame[label])))
        print(result)
        categoricals[label] = result
        data_frame[label] = data_frame[label].map(result)

    replace_empty(data_frame)
    data_after_process['result'] = data_frame
    data_after_process['map'] = categoricals
    return data_after_process


def map_labels( data_frame, categorical_labels, map):
    for label in categorical_labels:
        result = map[label]
        data_frame[label] = data_frame[label].map(result)
    return data_frame

def drop_labels( data_frame, ignore_labels):
    data_frame.drop(ignore_labels, axis = 1)
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

    data_frame.drop(ignore_labels)

    for label in categorical_labels:
        print(instanceName)
        print(label)
        result = get_unique_values(list(set(data_frame[label])))
        print(result)
        data_frame[label] = data_frame[label].map(result)

    replace_empty(data_frame)

    return data_frame
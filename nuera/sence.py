from os import walk
import sys

sys.path.append('../')
from sences.read_json import load_json_data
from sences.read_data_frame import read_csv
from centroid.process_data import get_data_model
from dendrites.preprocess_data import process
from axion_hillock.generate_responce import get_response_obj

process_path = "/home/chandra/PycharmProjects/nuera1/sences/nueron_data/crop/"
signals = {}
max_level = 0
# Iterate directory
for (dir_path, dir_names, file_names) in walk(process_path):
    for file_name in file_names:
        print(process_path + file_name)
        data = load_json_data(process_path + file_name)
        key = 'level_'+str(data['level'])
        signals[key] = signals.get(key, [])
        signals[key].append(data)
        if data['level'] > max_level:
            max_level = data['level']
print(signals)
for level in range(max_level):
    key = 'level_'+str(level)
    print(key)
    nuerons = signals.get(key, [])
    print(nuerons)
    for nueron in nuerons:
        train_data_pre = read_csv(nueron.get('train_data'))
        instance_name = nueron.get('instance_name')
        categorical_labels = nueron.get('categorical_labels')
        ignore_columns = nueron.get('ignore_columns')
        train_data = process(instance_name, train_data_pre, categorical_labels, ignore_columns)
        model_name = nueron.get('modal_name')
        result_column = nueron.get('result_column')
        additional_param = nueron.get('additional_model_parameters')
        model  = get_data_model(model_name, train_data, result_column, additional_param)
        test_data = read_csv(nueron.get('test_data'))
        result = model.predict(test_data)
        dynamic_result_data = nueron.get('dynamic_result_data')
        for dynamic_data in dynamic_result_data:
            file_name = nueron.get('result_data_name')
            result_columns = nueron.get('result_columns')
            mapping_features = nueron.get('mapping_features')
            result_data = nueron.get('result_data')
            target_data_frame= get_response_obj(file_name, train_data_pre, mapping_features, result_columns, result, result_data[0])
            target_data_frame.to_csv(file_name, index=False)
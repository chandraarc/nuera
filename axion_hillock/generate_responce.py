import sys
import pandas as pd

sys.path.append('../')
from sences.read_data_frame import read_csv


def get_response_obj(file_name, data_frame, mapping_columns, target_columns, result, result_column):
    target_data_frame = read_csv(file_name)
    map = {}
    target_columns.append(result_column)
    target_df = pd.DataFrame(columns=target_columns)

    for target_index, target_row in target_data_frame.iterrows():
        target_values = []
        for index, row in data_frame.iterrows():

            isMatched = True
            for map_column in mapping_columns:
                if target_row[map_column] != row[map_column]:
                    isMatched = False

            if isMatched and map[index] is None:
                for column in target_columns:
                    target_values.append(row[column])
                target_values.append(result[index])

        new_row = pd.Series(target_values, index=target_columns)
        target_df = target_df.append(new_row, ignore_index=True)

    return target_data_frame.merge(target_df, left_index=True, right_index=True)
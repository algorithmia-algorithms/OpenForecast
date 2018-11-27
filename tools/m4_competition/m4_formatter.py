import csv
import numpy as np
import pandas as pd
import json
import argparse


r"""
This dataset has a ton of great data, but some of it is missing. For many variables, there is a limited number of elements, and some are missing.
We need to figure out a threshold 
"""

def load_data_file(file_path):
    data = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data

def format_and_select(data, length, max_vars):
    in_tensor = np.asarray(data)[:, 1:]
    out_tensor = []
    headers = []
    for i in range(max_vars):
        variable = in_tensor[:, i]
        var_name = variable[0]
        var_data = variable[1:]
        var_data = trim_to_first_nan(var_data)
        if var_data.shape[0] >= length:
            header = {'index': i, 'header': var_name}
            var_data = var_data[0:length]
            out_tensor.append(var_data)
            headers.append(header)
    out_tensor = np.stack(out_tensor, axis=1)
    out_tensor = out_tensor.tolist()
    output = {'tensor': out_tensor, 'key_variables': headers[:3]}
    return output



def trim_to_first_nan(variable: np.ndarray):
    variable = pd.to_numeric(variable, errors='coerce')
    nans = np.isnan(variable)
    has_nans = nans.any()
    if has_nans:
        first_nan_index = np.where(nans == True)[0][0]
        output = variable[0:first_nan_index]
    else:
        output = variable
    return output

def serialize_to_file(path, object):
    with open(path, 'w') as f:
        json.dump(object, f)

if __name__ == "__main__":
    input_path = "/home/zeryx/algorithmia/research/m4/Yearly-train.csv"
    output_path = "/tmp/m4_yearly_0.1.json"
    num_vars = 5
    data = load_data_file(input_path)
    output = format_and_select(data, 1000, num_vars)
    serialize_to_file(output_path, output)
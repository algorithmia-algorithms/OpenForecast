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

def format_and_select(data, min_length):
    in_tensor = np.asarray(data)[:, 1:]
    out_tensor = []
    headers = []
    for i in range(in_tensor.shape[1]):
        variable = in_tensor[:, i]
        var_name = variable[0]
        var_data = variable[1:]
        var_data = trim_to_first_nan(var_data)
        if var_data.shape[0] >= min_length:
            header = {'index': i, 'header': var_name}
            out_tensor.append(var_data)
            headers.append(header)
    out_tensor = np.array(out_tensor)
    output = {'tensor': out_tensor.tolist(), 'headers': headers}
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



if __name__ == "__main__":
    input_path = "/tmp/m4_yearly.csv"
    output_path = "/tmp/m4_yearly_0.1.json"
    num_vars = 15
    data = load_data_file(input_path)
    format_and_select(data, 1500)
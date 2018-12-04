import csv
import numpy as np
import pandas as pd
import json
import argparse


def format_for_algorithm(data, length, max_vars):
    r"""
    * We limit the maximum number of variables to "max_vars", so even if most of our variables are
    longer than "sequence_length", we truncate the rest to keep the formatted dataset trim.
    * And finally, for our demo we are only selecting the first variable as a 'key_variable', you can change this
    as desired.
    """

    in_tensor = np.asarray(data)[1:, 1:]
    out_tensor = []
    for i in range(max_vars):
        variable = in_tensor[i, :]
        var_data = trim_to_first_nan(variable)
        if var_data.shape[0] >= length:
            var_data = var_data[0:length]
            out_tensor.append(var_data)
    if len(out_tensor) == 0:
        raise Exception('the requested sequence length is too long for your data, please select a smaller number.')
    else:
        out_tensor = np.stack(out_tensor, axis=1)
    out_tensor = out_tensor.tolist()
    output = {'tensor': out_tensor}
    return output



def trim_to_first_nan(variable):
    r"""
    This function uses `pandas` to find non-numeric characters (missing values, or invalid entries) for each variable.
    When a non-numeric character is found, the algorithm then trims the variable sequence  from 0 -> last numeric value.
    """

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

def load_data_file(file_path):
    data = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The m4 dataset formatter.")
    parser.add_argument('--input_path', type=str, help="The local system path to the m4 training dataset (in csv form), can be any type.")
    parser.add_argument('--output_path', type=str,
                        help="The local system path to where the formatted dataset should live.")
    parser.add_argument('--num_of_variables', type=int, help="The maximum number of variables we wish to track.")
    parser.add_argument('--sequence_length', type=int, help="The desired sequence length, shorter squences will be filtered out.")
    args = parser.parse_args()
    data = load_data_file(args.input_path)
    output = format_for_algorithm(data, args.sequence_length, args.num_of_variables)
    serialize_to_file(args.output_path, output)
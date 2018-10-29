import numpy as np
import torch


# During initial training, we check if a header is present, if so we preserve the headers in the model for variable description.
def process_input(data, multiplier):
    headers = data['headers']
    feature_columns = data['columns_to_forecast']
    tensor = data['tensor']
    tensor = np.asarray(tensor, dtype=np.float64)
    io_dims = tensor.shape[1]
    normalized_data, norm_boundaries = normalize_and_remove_outliers(tensor, io_dims, multiplier)
    return normalized_data, norm_boundaries, headers, feature_columns

# We first remove outliers based on the new dataset.
# However, we normalize based on the original training data.
# This is to make sure we're consistent in values fed into the network.
def normalize_and_remove_outliers(data, dimensions, multiplier, norm_boundaries=None):
    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    for i in range(dimensions):
        for j in range(len(data[:, i])):
            max_delta = mean[i] - multiplier * sd[i]
            if not (data[j, i] > max_delta):
                print('clipped {} for being too far above the mean.'.format(str(data[j, i])))
                data[j, i] = max_delta
            elif not (-data[j, i] > max_delta):
                print('clipped {} for being too far below the mean.'.format(str(data[j, i])))
                data[j, i] = -max_delta
    if norm_boundaries:
        for i in range(dimensions):
            numerator = np.subtract(data[:, i], norm_boundaries[i]['min'])
            data[:, i] = np.divide(numerator, norm_boundaries[i]['max'] - norm_boundaries[i]['min'])

    else:
        norm_boundaries = list()
        for i in range(dimensions):
            max = np.max(data[:, i], axis=0)
            min = np.min(data[:, i], axis=0)
            norm_boundaries.append({'max': max, 'min': min})
            numerator = np.subtract(data[:, i], min)
            data[:, i] = np.divide(numerator, max - min)
    return data, norm_boundaries

# used for reverting the normalization process for forecasts. The "norm_boundaries" variables are defined in
# normalize_and_remove_outliers if no 'norm_boundaries' variable is provided.
def revert_normalization(data, state):
    norm_boundaries = state['norm_boundaries']
    io_shape = state['io_width']
    output = np.empty(data.shape, float)
    for i in range(io_shape):
        min = norm_boundaries[i]['min']
        max = norm_boundaries[i]['max']
        multiplier = max-min
        intermediate = np.multiply(data[:, i], multiplier)
        result = np.add(intermediate, min)
        output[:, i] = result
    return output


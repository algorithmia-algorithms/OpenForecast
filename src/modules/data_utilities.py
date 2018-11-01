import numpy as np

from uuid import uuid4
import matplotlib.pyplot as plt
from src.modules import network
r"""
This function does a lot of stuff
- First, it extracts some data from the incoming data object, specifically the 'tensor', 
  'headers', and 'columns_to_forecast' variables.
- Second, we populate the 'meta_data' object if it hasn't been initialized already
- And finally at the same time, we define the model architecture via the model_complexity variable, and define tensor shapes.
"""

def process_input(data: dict, parameters, meta_data: dict):
    tensor = data['tensor']
    tensor = np.asarray(tensor, dtype=np.float64)
    meta_data['training_time'] = meta_data.get('training_time', parameters.training_time)
    meta_data['headers'] = meta_data.get('headers', data['headers'])
    meta_data['feature_columns'] = meta_data.get('feature_columns', data['columns_to_forecast'])
    meta_data['io_dimension'] = meta_data.get('io_dimension', tensor.shape[1])
    meta_data['norm_boundaries'] = meta_data.get('norm_boundaries', calc_norm_boundaries(tensor, meta_data['io_dimension']))
    normalized_data = normalize_and_remove_outliers(tensor, parameters.outlier_removal_multiplier, meta_data)

    meta_data['architecture'] = define_architecture(parameters.model_complexity,
                                                    meta_data['feature_columns'], meta_data['io_dimension'], parameters.io_noise)
    meta_data['tensor_shape'] = {'memory': (meta_data['architecture']['recurrent']['depth'], 1, meta_data['architecture']['recurrent']['output']),
                                 'residual': (1, 1, meta_data['architecture']['recurrent']['output'])}

    meta_data['forecast_length'] = parameters.forecast_length
    meta_data['complexity'] = parameters.model_complexity
    meta_data['io_noise'] = parameters.io_noise
    return normalized_data, meta_data


r"""
By using the complexity parameter (which has a range between 0.0 and 1.0), we're able to figure out how many parameters
our recurrent neural network architecture should have
"""

def define_architecture(complexity: float, feature_columns: list, io_dimensions: int, io_noise: float):
    architecture = dict()
    architecture['gaussian_noise'] = {}
    architecture['linear_in'] = {}
    architecture['linear_out'] = {}
    architecture['recurrent'] = {}
    architecture['gaussian_noise']['stddev'] = io_noise
    min_depth = 1
    max_depth = 5
    min_mem_width, min_lin_width = int(10 * len(feature_columns)), int(10 * len(feature_columns))
    max_mem_width, max_lin_width = int(100 * len(feature_columns)), int(100 * len(feature_columns))
    depth = int(complexity * (max_depth - min_depth)) + min_depth
    linear_width = int(complexity * (max_lin_width - min_lin_width)) + min_lin_width
    memory_width = int(complexity * (max_mem_width - min_mem_width)) + min_mem_width

    architecture['linear_in']['input'] = io_dimensions
    architecture['linear_out']['output'] = io_dimensions
    architecture['linear_out']['input'] = memory_width
    architecture['linear_in']['output'] = linear_width
    architecture['recurrent']['input'] = linear_width
    architecture['recurrent']['output'] = memory_width
    architecture['recurrent']['depth'] = depth

    return architecture


# We first remove outliers based on the new dataset.
# However, we normalize based on the original training data.
# This is to make sure we're consistent in values fed into the network.
def normalize_and_remove_outliers(data: np.ndarray, multiplier: float, meta_data: dict):
    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    dimensions = meta_data['io_dimension']
    norm_boundaries = meta_data['norm_boundaries']
    for i in range(dimensions):
        for j in range(len(data[:, i])):
            max_delta = mean[i] - multiplier * sd[i]
            if not (data[j, i] > max_delta):
                print('clipped {} for being too far above the mean.'.format(str(data[j, i])))
                data[j, i] = max_delta
            elif not (-data[j, i] > max_delta):
                print('clipped {} for being too far below the mean.'.format(str(data[j, i])))
                data[j, i] = -max_delta
    for i in range(dimensions):
        numerator = np.subtract(data[:, i], norm_boundaries[i]['min'])
        data[:, i] = np.divide(numerator, norm_boundaries[i]['max'] - norm_boundaries[i]['min'])

    return data

def calc_norm_boundaries(data: np.ndarray, dimensions: int):
    norm_boundaries = list()
    for i in range(dimensions):
        max = np.max(data[:, i], axis=0)
        min = np.min(data[:, i], axis=0)
        norm_boundaries.append({'max': max, 'min': min})
    return norm_boundaries


# Used for reverting the normalization process for forecasts.
# The "norm_boundaries" variables are defined in the meta_data object.
# maps the tensors values back to the original representation
def revert_normalization(data: np.ndarray, meta_data: dict):
    norm_boundaries = meta_data['norm_boundaries']
    features = meta_data['feature_columns']
    output = np.empty(data.shape, float)
    for i in range(len(features)):
        min = norm_boundaries[features[i]]['min']
        max = norm_boundaries[features[i]]['max']
        multiplier = max-min
        intermediate = np.multiply(data[:, i], multiplier)
        result = np.add(intermediate, min)
        output[:, i] = result
    return output


# Prepares the raw output from our forecast operation for export to the user.
# uses the variable Headers to label the dimension appropriately.
def format_forecast(forecast: np.ndarray, meta_data: dict):
    true_forecast = revert_normalization(forecast, meta_data)
    headers = meta_data['headers']
    result = dict()
    for i in range(len(headers)):
        result[headers[i]] = true_forecast[:, i].tolist()
    return result


# Uses matplotlib to create a graph of the forecast tensor, useful for visualizing the results.
def generate_graph(x: np.ndarray, forecast: np.ndarray, meta_data: dict):
    headers = meta_data['headers']

    forecast_length = forecast.shape[0]
    seq_length = x.shape[0]
    filename = '/tmp/{}.png'.format(str(uuid4()))
    if forecast_length >= seq_length:
        raise network.AlgorithmError("requested forecast length for graphing,"
                                     " beacuse input sequence is {} long".format(str(seq_length)))
    x = np.arange(1, forecast_length*2+1)
    for i in range(len(headers)):
        label = headers[i]
        plt.plot(x[-forecast_length:], forecast[:, i], linestyle='--', label=label)
    plt.savefig(filename)
    plt.close()
    return filename

def save_graph(graph_path, remote_url):
    return network.put_file(graph_path, remote_url)


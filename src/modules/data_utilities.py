import numpy as np
from uuid import uuid4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.modules import network_utilities



# This catch-all function extracts key parameters from your dataset, and if your `meta_data` object is not defined, populates
# it with data collected from the dataset, and the input parameters object.
# If meta_data is already defined, most of those steps are skipped.
# TODO: clean this up, potentially break out meta_data processing into it's own function.
def process_input(data: dict, parameters, meta_data: dict = None):
    tensor = data['tensor']
    tensor = np.asarray(tensor, dtype=np.float64)
    if meta_data:
        if parameters.forecast_length:
            meta_data['forecast_length'] = parameters.forecast_length
    else:
        meta_data = dict()
        meta_data['training_time'] = parameters.training_time
        meta_data['key_variables'] = data['key_variables']
        meta_data['io_dimension'] = tensor.shape[1]
        meta_data['norm_boundaries'] = calc_norm_boundaries(tensor, meta_data['io_dimension'])

        new_architecture = define_network_geometry(parameters.model_complexity,
                                                   len(meta_data['key_variables']),
                                                   meta_data['io_dimension'])
        tensor_shape = {'memory': (new_architecture['recurrent']['depth'],
                                   1, new_architecture['recurrent']['output']),
                        'residual': (1, 1, new_architecture['recurrent']['output'])}
        meta_data['architecture'] = new_architecture
        meta_data['tensor_shape'] = tensor_shape
        meta_data['complexity'] = parameters.model_complexity
        meta_data['io_noise'] = parameters.io_noise
        meta_data['forecast_length'] = parameters.forecast_length
    normalized_data = normalize_and_remove_outliers(tensor, parameters.outlier_removal_multiplier, meta_data)
    return normalized_data, meta_data


#    This function takes your complexity parameters, and other things that define your dataset, to automatically generate
#    the width of certain types of layers, and the number of layers in your recurrent module.
#    TODO: `complexity` right now only effects recurrent module depth, but it should influence layer width as well.
def define_network_geometry(complexity: float, y_width: int, io_dimensions: int):
    architecture = dict()
    architecture['linear_in'] = {}
    architecture['linear_out'] = {}
    architecture['recurrent'] = {}
    min_depth = 1
    max_depth = 5
    min_mem_width, min_lin_width = int(1 * y_width), int(1 * y_width)
    max_mem_width, max_lin_width = int(10 * y_width), int(10 * y_width)
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
    features = meta_data['key_variables']
    output = np.empty(data.shape, float)
    for i in range(len(features)):
        index = features[i]['index']
        min = norm_boundaries[index]['min']
        max = norm_boundaries[index]['max']
        multiplier = max-min
        intermediate = np.multiply(data[:, i], multiplier)
        result = np.add(intermediate, min)
        output[:, i] = result
    return output


# Prepares the raw output from our forecast operation for export to the user.
# uses the variable Headers to label the dimension appropriately.
def format_forecast(forecast: np.ndarray, meta_data: dict):
    true_forecast = revert_normalization(forecast, meta_data)
    feature_columns = meta_data['key_variables']
    result = dict()
    for i in range(len(feature_columns)):
        header = feature_columns[i]['header']
        result[header] = true_forecast[:, i].tolist()
    return result


# Uses matplotlib to create a graph of the forecast tensor, useful for visualizing the results.
def generate_graph(x: np.ndarray, forecast: np.ndarray, meta_data: dict):
    feature_columns = meta_data['key_variables']

    forecast_length = forecast.shape[0]
    seq_length = x.shape[0]
    filename = '/tmp/{}.png'.format(str(uuid4()))
    if forecast_length >= seq_length:
        raise network_utilities.AlgorithmError("requested forecast length for graphing,"
                                     " beacuse input sequence is {} long".format(str(seq_length)))
    x = np.arange(1, forecast_length*2+1)
    for i in range(len(feature_columns)):
        label = feature_columns[i]['header']
        plt.plot(x[-forecast_length:], forecast[:, i], linestyle='--', label=label)
    plt.savefig(filename)
    plt.close()
    return filename

def save_graph(graph_path, remote_url):
    return network_utilities.put_file(graph_path, remote_url)


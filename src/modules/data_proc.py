import numpy as np
from src.modules import misc
from ergonomics.serialization import save_portable, load_portable
from uuid import uuid4
import math
def is_header(row):
    try:
        _ = float(row[0])
        return False
    except:
        return True


# For incremental training, we know what the header names are already so lets just pop them if they exist.
def process_sequence_incremental(data, state, multiplier):
    if is_header(data[0]):
        data.pop(0)
    step_size = state['step_size']
    objective = []
    for i in range(0, len(data), step_size):
        objective.append(data[i])
    data = np.asarray(objective).astype(np.float)
    shape = data.shape
    norms = state['norm_boundaries']
    io_width = state['io_width']
    if shape[1] != io_width:
        raise misc.AlgorithmError("data dimensions are different from expected, got {}\t expected {}.".format(str(shape[1]), str(io_width)))
    data, _ = normalize_and_remove_outliers(data, io_width, multiplier, norms)
    x, y = prepare_x_y(data)
    data = {'x': x, 'y': y}
    return data

# During initial training, we check if a header is present, if so we preserve the headers in the model for variable description.
def process_sequence_initial(data, multiplier):
    if is_header(data[0]):
        headers = data.pop(0)
    else:
        headers = np.arange(len(data[0]), dtype=str).tolist()
    floated = []
    if len(data) >= 5000:
        step_size = int(math.ceil(len(data)/5000))
        objective = []
        for i in range(0, len(data), step_size):
            objective.append(data[i])
    else:
        step_size = 1
        objective = data
    for elm in objective:
        new_dim = []
        for dim in elm:
            new_dim.append(float(dim))
        floated.append(new_dim)
    npdata = np.asarray(floated).astype(np.float)
    io_dims = npdata.shape[1]
    print(io_dims)
    normalized_data, norm_boundaries = normalize_and_remove_outliers(npdata, io_dims, multiplier)
    x, y = prepare_x_y(normalized_data)
    output = {'x': x, 'y': y}
    return output, norm_boundaries, headers, step_size


def prepare_x_y(data):
    beam_width = 1
    x = data[:-(beam_width+1)]
    y = []
    for i in range(1, beam_width+1):
        y_i = data[i:-(beam_width+1 - i)]
        y.append(y_i)
    y = np.asarray(y).astype(np.float)
    return x, y


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


def get_frame(remote_path):
    local_file = misc.get_data(remote_path)
    with open(local_file) as f:
        lines = f.read().split('\n')
        csv = [x.split(',') for x in lines]
    csv.pop(-1)
    return csv


def save_network_to_algo(network, remote_file_path):
    network_def_path = "src.modules.net_def"
    local_file_path = "/tmp/{}".format(str(uuid4()))
    save_portable(network, network_def_path, local_file_path)
    misc.put_file(local_file_path, remote_file_path)
    return remote_file_path


def load_network_from_algo(remote_file_path):
    local_file_path = misc.get_data(remote_file_path)
    network = load_portable(local_file_path)
    state = network.get_state()
    return network, state


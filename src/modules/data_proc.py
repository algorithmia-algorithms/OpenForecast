import numpy as np
from src.modules.misc import AlgorithmError, get_file

def process_frames_incremental(data, state, multiplier):
    data = np.asarray(data).astype(np.float)
    shape = data.shape
    beam_width = state['lookup_beam_width']
    norms = state['norm_boundaries']
    io_width = state['io_width']
    if shape[1] != io_width:
        raise AlgorithmError("data dimensions are different from expected, got {}\t expected {}.".format(str(shape[1]), str(io_width)))
    data, _ = normalize_and_remove_outliers(data, io_width, multiplier, norms)
    x, y = prepare_x_y(data, beam_width)
    data = {'x': x, 'y': y}
    return data


def process_frames_initial(data, multiplier, beam_width):
    data = np.asarray(data).astype(np.float)
    io_dims = data.shape[1]
    data, norm_boundaries = normalize_and_remove_outliers(data, io_dims, multiplier)
    x, y = prepare_x_y(data, beam_width)
    data = {'x': x, 'y': y}
    return data, norm_boundaries


def prepare_x_y(data, beam_width):
    x = data[:-(beam_width+1)]
    y = []
    for i in range(1, beam_width+1):
        y_i = data[i:-(beam_width+1 - i)]
        y.append(y_i)
    y = np.asarray(y).astype(np.float)
    return x, y


def normalize_and_remove_outliers(data, dimensions, multiplier, norm_boundaries=list()):
    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    for i in range(dimensions):
        for j in range(len(data[:, i])):
            if not (data[j, i] > mean[i] - multiplier * sd[i]):
                print('popped {} for being out of bounds.'.format(str(data[j, i])))
                data[j, i] = False

    if norm_boundaries == list():
        for i in range(dimensions):
            max = np.max(data[:, i])
            min = np.min(data[:, i])
            norm_boundaries.append({'max': max, 'min': min})
    for i in range(dimensions):
        numerator = np.subtract(data[:, i], norm_boundaries[i]['min'])
        data[:, i] = np.divide(numerator, norm_boundaries[i]['max'] - norm_boundaries[i]['min'])
    return data, norm_boundaries

# used for reverting the normalization process for forecasts
def revert_normalization(data, state):
    norm_boundaries = state['norm_boundaries']
    io_shape = state['io_width']
    for i in range(io_shape):
        min = norm_boundaries[i]['min']
        max = norm_boundaries[i]['max']
        data[:, i] = np.multiply(data[:, i], (max - min))
        data[:, i] = np.add(data[:, i], min)
    return data


def get_frame(remote_path):
    local_file = get_file(remote_path)
    with open(local_file) as f:
        lines = f.read().split('\n')
        csv = [x.split(',') for x in lines]
    return csv
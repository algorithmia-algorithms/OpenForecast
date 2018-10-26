import copy
from math import isnan
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from src.modules.net_def import model
from src.modules.graph import test_graph
from . import data_utilities
cuda = False

def create_forecasts(data_frame, network, state,  number_of_forecasts, future_length,):

    init_true = state['true_history']
    init_pred = state['predicted_history']
    init_h = state['act1_h']
    io_width = state['io_width']
    if data_frame:
        criterion = nn.MSELoss()
        input = Variable(torch.from_numpy(data_frame['x']), requires_grad=False).float()
        length = input.shape[0]
        if cuda:
            input = input.cuda()
        pred_1, true_1, hidden_h, _ = network.forward(input, length, init_true, init_pred, init_h)
        state = network.get_state()
        update_loss = criterion(pred_1, true_1)
        update_loss = update_loss.cpu().data.numpy()
        print("model update loss: {}".format(str(update_loss)))
    else:
        pred_1 = init_pred
        hidden_h = init_h
    print("checkpoint state loaded, beginning to forecast")
    normal_forecasts = np.empty((number_of_forecasts, future_length, io_width), dtype=np.float)
    raw_forecasts = np.empty((number_of_forecasts, future_length, io_width), dtype=np.float)
    for i in range(number_of_forecasts):
        result = network.forecast(future_length=future_length, historical_data=pred_1, memory=hidden_h).cpu().data.numpy()
        denormalized = data_utilities.revert_normalization(result, state)
        normal_forecasts[i] = result
        raw_forecasts[i] = denormalized
        print("forecast {} complete".format(str(i)))
    return normal_forecasts, raw_forecasts, state




def train_autogenerative_model(data_frame, network, state, meta_data):
    input = Variable(torch.from_numpy(data_frame), requires_grad=False).float()
    criterion = nn.MSELoss()
    best_loss = 10000000
    parameters = network.parameters()
    optimizer = optim.Adam(parameters, lr=35e-4)
    if 'checkpoint' in state:
        checkpoint_memory = state['checkpoint']['memory_layers']
        checkpoint_residual = state['checkpoint']['residual_channels']
    else:
        checkpoint_memory = state['init']['memory_layers']
        checkpoint_residual = state['init']['residual_channels']

    trainables, targetables = data_utilities.segment_data(input, meta_data['forecast_length'])
    number_of_subsequences = len(trainables)
    start = perf_counter()
    total_time = 0
    while True:
        if total_time >= meta_data['training_time']:
            break
        optimizer.zero_grad()
        residual_t, memory_t = clone_state(checkpoint_residual, checkpoint_memory)
        loss = 0
        predictions = []
        for i in range(number_of_subsequences):
            training_t = trainables[i]
            target_t = targetables[i]
            forecast_t = torch.zeros(target_t.shape)
            input_length = training_t.shape[0]
            residual_t, memory_t = network.forward(training_t, residual_t, memory_t, input_length)
            residual_forecast, memory_forecast = clone_state(residual_t, memory_t)
            forecast_t = network.forecast(meta_data['forecast_length'], residual_forecast, memory_forecast, training_t[-1], forecast_t)
            # forecast_t = torch.stack(forecast_t)
            predictions.append(forecast_t)
            loss_t = criterion(forecast_t, target_t)
            loss += loss_t
        loss /= number_of_subsequences
        loss_cpu = loss.item()
        print('training loss: {}'.format(str(loss_cpu)))
        loss.backward()
        optimizer.step()
        total_time = perf_counter() - start
    best_predictions = torch.stack(predictions)
    graphable_targets = targetables.view(targetables.shape[0]*targetables.shape[1], targetables.shape[2]).numpy()
    graphable_predictions = best_predictions.view(best_predictions.shape[0]*best_predictions.shape[1], best_predictions.shape[2]).numpy()
    test_graph(graphable_predictions, graphable_targets)
    updated_residual, updated_memory = network.forward(input, checkpoint_residual, checkpoint_memory, input.shape[0])
    print('best overall training loss: {}'.format(str(best_loss)))
    state['checkpoint']['memory_layers'] = updated_memory.numpy.tolist()
    state['checkpoint']['residual_channels'] = updated_residual.numpy.tolist()
    return best_loss, network, state

# this determines the learning rate based on comparing the prime length to the current incremental length
def determine_lr(data, state):
    incr_length = len(data)
    ratio = float(incr_length) / state['prime_length']
    incr_lr = ratio*state['prime_lr']
    return incr_lr


def clone_state(residual, memory):
    out_res = residual.clone()
    out_m = memory.clone()
    return out_res, out_m

def generate_state(num_layers, width):
    memory = torch.zeros(num_layers, 1, width)
    residual = torch.zeros(1, 1, width)
    return residual, memory

def define_architecture(complexity, feature_columns):

    min_depth = 1
    max_depth = 5
    min_mem_width, min_lin_width = int(10 * len(feature_columns)), int(10 * len(feature_columns))
    max_mem_width,max_lin_width = int(100 * len(feature_columns)), int(100 * len(feature_columns))
    depth = int(complexity*(max_depth - min_depth)) + min_depth
    linear_width = int(complexity*(max_lin_width - min_lin_width)) + min_lin_width
    memory_width = int(complexity*(max_mem_width - min_mem_width)) + min_mem_width
    return depth, memory_width, linear_width


def init_network(io_width, io_noise, complexity, feature_columns, headers):
    depth, memory_width, linear_width = define_architecture(complexity, feature_columns)

    network = model.Model(io_width=io_width, io_noise=io_noise,
                          memory_width=memory_width, linear_width=linear_width, depth=depth).float()
    residuals, memory_layers = generate_state(depth, memory_width)
    state = dict()
    meta_data = dict()
    state['init'] = {'memory_layers': memory_layers,
                     'residual_channels': residuals
                     }
    meta_data['forecast_length'] = 10
    meta_data['complexity'] = complexity
    meta_data['io_width'] = io_width
    meta_data['feature_columns'] = feature_columns
    meta_data['headers'] = headers
    meta_data['prime_length'] = None
    meta_data['norm_boundaries'] = None
    return network, meta_data, state

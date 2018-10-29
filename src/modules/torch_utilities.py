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


#
# def create_forecasts(data_frame, network, state,  number_of_forecasts, future_length):
#
#     init_true = state['true_history']
#     init_pred = state['predicted_history']
#     init_h = state['act1_h']
#     io_width = state['io_width']
#     if data_frame:
#         criterion = nn.MSELoss()
#         input = Variable(torch.from_numpy(data_frame['x']), requires_grad=False).float()
#         length = input.shape[0]
#         if cuda:
#             input = input.cuda()
#         pred_1, true_1, hidden_h, _ = network.process(input, length, init_true, init_pred, init_h)
#         state = network.get_state()
#         update_loss = criterion(pred_1, true_1)
#         update_loss = update_loss.cpu().data.numpy()
#         print("model update loss: {}".format(str(update_loss)))
#     else:
#         pred_1 = init_pred
#         hidden_h = init_h
#     normal_forecasts = np.empty((number_of_forecasts, future_length, io_width), dtype=np.float)
#     raw_forecasts = np.empty((number_of_forecasts, future_length, io_width), dtype=np.float)
#     for i in range(number_of_forecasts):
#         result = network.forecast(future_length=future_length, historical_data=pred_1, memory=hidden_h).cpu().data.numpy()
#         denormalized = data_utilities.revert_normalization(result, state)
#         normal_forecasts[i] = result
#         raw_forecasts[i] = denormalized
#         print("forecast {} complete".format(str(i)))
#     return normal_forecasts, raw_forecasts, state

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
    forecast_length = meta_data['forecast_length']
    feature_columns = meta_data['feature_columns']
    x, y = segment_data(input, forecast_length)
    start = perf_counter()
    total_time = 0
    while True:
        if total_time >= meta_data['training_time']:
            break
        optimizer.zero_grad()
        residual_t, memory_t = clone_state(checkpoint_residual, checkpoint_memory)
        loss = 0
        graphable_preds = []
        graphable_targets = []
        for i in range(len(x)):
            training_t = x[i].view(1, -1)
            target_t = y[i].view(forecast_length, -1)
            forecast_t, residual_t, memory_t = step(network, training_t, residual_t, memory_t, forecast_length)
            target_f, forecast_f = filter_feature_cols(target_t, forecast_t, feature_columns)
            graphable_preds.append(forecast_f)
            graphable_targets.append(target_f)
            loss += criterion(forecast_f, target_f)
        loss /= len(x)
        loss_cpu = loss.item()
        print('training loss: {}'.format(str(loss_cpu)))
        # Selecting only the first prediction/target for each step
        graphable_preds = torch.stack(graphable_preds).detach().numpy()
        graphable_targets = torch.stack(graphable_targets).detach().numpy()
        graphable_preds = graphable_preds[:, 1, :]
        graphable_targets = graphable_targets[:, 1, :]
        if int(total_time) % 15 == 0:
            test_graph(graphable_targets, graphable_preds)
        loss.backward()
        optimizer.step()
        total_time = perf_counter() - start
        print("current training time: {}s".format(str(total_time)))
    updated_residual, updated_memory = network.process(input, checkpoint_residual, checkpoint_memory, input.shape[0])
    print('best overall training loss: {}'.format(str(best_loss)))
    state['checkpoint']['memory_layers'] = updated_memory.numpy.tolist()
    state['checkpoint']['residual_channels'] = updated_residual.numpy.tolist()
    return best_loss, network, state

# At every step, we forecast the next n points.
def step(network, training_t: torch.Tensor, residual_t: torch.Tensor, memory_t: torch.Tensor,
         forecast_length: int):
    io_width = training_t.shape[1]
    output_t, residual_t, memory_t = network.forward(training_t, residual_t, memory_t)

    forecast_tensor = forecast_step(network, output_t, residual_t, memory_t, forecast_length, io_width)

    return forecast_tensor, residual_t, memory_t


def forecast_step(network, x_t, residual_t, memory_t,  forecast_length, io_width):
    residual_forecast, memory_forecast = clone_state(residual_t, memory_t)
    forecast_tensor = torch.zeros(forecast_length, io_width)
    forecast_tensor[0] = x_t
    for i in range(1, forecast_length):
        output, residual_forecast, memory_forecast = network.forward(forecast_tensor[i-1], residual_forecast,
                                                                     memory_forecast)
        forecast_tensor[i] = output
    return forecast_tensor

def filter_feature_cols(targets, forecasts, feature_columns):
    if feature_columns:
        filtered_targets = []
        filtered_forecasts = []
        for feature in feature_columns:
            filtered_targets.append(targets[:, feature])
            filtered_forecasts.append(forecasts[:, feature])
        filtered_targets = torch.stack(filtered_targets, dim=1)
        filtered_forecasts = torch.stack(filtered_forecasts, dim=1)
    else:
        filtered_targets = targets
        filtered_forecasts = forecasts
    return filtered_targets, filtered_forecasts


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


def init_network(io_width, io_noise, complexity, feature_columns, headers, forecast_length):
    depth, memory_width, linear_width = define_architecture(complexity, feature_columns)

    network = model.Model(io_width=io_width, io_noise=io_noise,
                          memory_width=memory_width, linear_width=linear_width, depth=depth).float()
    residuals, memory_layers = generate_state(depth, memory_width)
    state = dict()
    meta_data = dict()
    state['init'] = {'memory_layers': memory_layers,
                     'residual_channels': residuals
                     }
    meta_data['forecast_length'] = forecast_length
    meta_data['complexity'] = complexity
    meta_data['io_width'] = io_width
    meta_data['feature_columns'] = feature_columns
    meta_data['headers'] = headers
    meta_data['prime_length'] = None
    meta_data['norm_boundaries'] = None
    return network, meta_data, state


def segment_data(data: torch.Tensor, forecast_length: int):
    segments = []
    for i in range(data.shape[0]-(forecast_length+1)):
        segment = data[i+1:i+forecast_length+1]
        segments.append(segment)
    # num_segments = len(segments)-1
    # x_seg = segments[0:-1]
    # y_seg = segments[1:]
    #
    # x = torch.zeros(num_segments, forecast_length, dims, requires_grad=False).float()
    # y = torch.zeros(num_segments, forecast_length, dims, requires_grad=False).float()
    # for i in range(num_segments):
    #     x[i] = x_seg[i]
    #     y[i] = y_seg[i]
    x = data[:-(forecast_length+1)]
    y = torch.stack(segments)
    return x, y
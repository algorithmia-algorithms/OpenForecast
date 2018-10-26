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




def train_autogenerative_model(data_frame, network, state, iterations):
    input = Variable(torch.from_numpy(data_frame), requires_grad=False).float()
    forecast_length = 25
    criterion = nn.MSELoss()
    best_loss = 10000000
    best_state = None
    parameters = network.parameters()
    optimizer = optim.Adam(parameters, lr=35e-4)
    print("ready to train")
    start_time = perf_counter()
    cur_iter = 0
    if 'checkpoint' in state:
        checkpoint_true = state['checkpoint']['history']
        checkpoint_memory = state['checkpoint']['memory_layers']
        checkpoint_residual = state['checkpoint']['residual_channels']
    else:
        checkpoint_true = state['init']['history']
        checkpoint_memory = state['init']['memory_layers']
        checkpoint_residual = state['init']['residual_channels']

    trainables, targetables = data_utilities.segment_data(input, forecast_length)
    number_of_subsequences = len(trainables)
    best_predictions = []
    while True:
        if cur_iter >= iterations:
            break
        time_delta = perf_counter()
        optimizer.zero_grad()
        residual_t, memory_t = clone_state(checkpoint_residual, checkpoint_memory)

        true_t = checkpoint_true.clone()
        loss = 0
        predictions = []
        for i in range(number_of_subsequences):
            training_t = trainables[i]
            target_t = targetables[i]
            input_length = training_t.shape[0]
            forward_step = i*forecast_length
            forecast_step = forward_step + forecast_length
            true_t, residual_t, hidden_t = network.forward(training_t, true_t, residual_t, memory_t, input_length, forward_step)
            residual_forecast, memory_forecast = clone_state(residual_t, memory_t)
            prediction_t = network.forecast(forecast_length, true_t, residual_forecast, memory_forecast, forecast_step)
            predictions.append(prediction_t.detach())
            loss_t = criterion(prediction_t, target_t)
            loss += loss_t
        loss /= number_of_subsequences
        loss_cpu = loss.item()
        if loss_cpu <= best_loss:
            best_loss = loss_cpu
            best_state = copy.deepcopy(network.state_dict())
            best_predictions = predictions
            cur_iter += 1
            print('current best pre-training loss: {}'.format(str(best_loss)))
        elif loss_cpu > best_loss*10 or isnan(loss_cpu):
            print("loss was {}\nover threshold, reloading to best state".format(str(loss_cpu)))
            network.load_state_dict(best_state)
            optimizer.zero_grad()
        else:
            cur_iter += 1
            print('training loss: {}'.format(str(loss_cpu)))
        loss.backward()
        print('total time: {}'.format(str(perf_counter() - start_time)))
        optimizer.step()
        print('delta time: {}'.format(str(perf_counter() - time_delta)))
    best_predictions = torch.stack(best_predictions)
    graphable_targets = targetables.view(targetables.shape[0]*targetables.shape[1], targetables.shape[2]).numpy()
    graphable_predictions = best_predictions.view(best_predictions.shape[0]*best_predictions.shape[1], best_predictions.shape[2]).numpy()
    test_graph(graphable_predictions, graphable_targets)
    updated_history, updated_residual, updated_memory = network.forward(input, checkpoint_true, checkpoint_residual, checkpoint_memory, input.shape[0], 0)
    print('best overall training loss: {}'.format(str(best_loss)))
    state['checkpoint']['history'] = updated_history.numpy().tolist()
    state['checkpoint']['memory_layers'] = updated_memory.numpy.tolist()
    state['checkpoint']['residual_channels'] = updated_residual.numpy.tolist()
    return best_loss, network, state

# this determines the learning rate based on comparing the prime length to the current incremental length
def determine_lr(data, state):
    incr_length = len(data)
    ratio = float(incr_length) / state['prime_length']
    incr_lr = ratio*state['prime_lr']
    return incr_lr


def clone_state(residual, hidden):
    out_res = []
    out_h = []
    for i in range(len(residual)):
        res = residual[i].clone()
        hid = hidden[i].clone()
        out_res.append(res)
        out_h.append(hid)
    return out_res, out_h

def generate_state(num_layers, width):
    memory_layers = []
    residual_channels  = []
    for _ in range(num_layers):
        memory = torch.zeros(1, 1, width)
        residual = torch.zeros(1, 1, width)
        memory_layers.append(memory)
        residual_channels.append(residual)
    residual_channels = tuple(residual_channels)
    memory_layers = tuple(memory_layers)
    return residual_channels, memory_layers


def initialize_network(io_width, width, depth, max_history, initial_lr, lr_multiplier, io_noise, perplexity,
                       attention_beam_width, headers):
    network = model.Model(memory_width=width, depth=depth,
                          io_width=io_width, io_noise=io_noise,
                          linear_width=attention_beam_width).float()
    true_history = torch.zeros(max_history, io_width, requires_grad=False).float()
    residuals, memory_layers = generate_state(depth, width)
    state = {}
    state['init'] = {'history': true_history,
                     'memory_layers': memory_layers,
                     'residual_channels': residuals
                     }

    state['perplexity'] = perplexity
    state['max_history'] = max_history
    state['io_width'] = io_width
    state['linear_width'] = attention_beam_width
    state['prime_lr'] = initial_lr
    state['lr_mul'] = lr_multiplier
    state['headers'] = headers
    state['prime_length'] = None
    state['norm_boundaries'] = None
    state['step_size'] = None
    return network, state

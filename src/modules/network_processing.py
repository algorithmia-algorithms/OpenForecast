import copy
from math import isnan
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from src.modules.net_def import actrnn
from . import data_proc
cuda = False

def create_forecasts(data_frame, network, state,  number_of_forecasts, future_length, noise_amount):

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
        result = network.forecast(future_length=future_length, pred_history=pred_1, hidden=hidden_h).cpu().data.numpy()
        denormalized = data_proc.revert_normalization(result, state)
        normal_forecasts[i] = result
        raw_forecasts[i] = denormalized
        print("forecast {} complete".format(str(i)))
    return normal_forecasts, raw_forecasts, state




def train_autogenerative_model(data_frame, network, state, iterations):
    input = Variable(torch.from_numpy(data_frame['x']), requires_grad=False).float()
    input_length = input.shape[0]
    criterion = nn.MSELoss()
    best_loss = 10000000
    best_state = None
    # optimizer = optim.Adadelta(network.parameters())
    optimizer = optim.Adam(network.parameters(), lr=35e-4)
    print("ready to train")
    start_time = perf_counter()
    cur_iter = 0
    tau = 7e-5
    if 'checkpoint' in state:
        working_true = state['checkpoint']['true_history']
        working_pred = state['checkpoint']['predicted_history']
        working_h = state['checkpoint']['act1_h']
    else:
        working_true = state['init']['true_history']
        working_pred = state['init']['predicted_history']
        working_h = state['init']['act1_h']

    while True:
        if cur_iter >= iterations:
            break
        time_delta = perf_counter()
        optimizer.zero_grad()
        output, target, hidden, ponder_cost = network.forward(input, input_length, working_true, working_pred, working_h)
        loss = criterion(output, target)
        ponder_loss = ponder_cost * tau
        loss_hat = loss + ponder_loss
        loss_cpu = loss_hat.item()
        print("ponder loss: {}".format(str(ponder_loss.data.numpy()[0])))
        print("non-ponder loss: {}".format(str(loss.data.numpy())))
        if loss_cpu <= best_loss:
            best_loss = loss_cpu
            best_state = copy.deepcopy(network.state_dict())
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
    final_pred, final_true, final_h, _ = network.forward(input, input_length, working_true, working_pred, working_h)
    print('best overall training loss: {}'.format(str(best_loss)))
    return best_loss, network, final_pred, final_true, final_h

# this determines the learning rate based on comparing the prime length to the current incremental length
def determine_lr(data, state):
    incr_length = len(data['x'])
    ratio = float(incr_length) / state['prime_length']
    incr_lr = ratio*state['prime_lr']
    return incr_lr




def initialize_network(io_width, hidden_width, hidden_depth, max_history, initial_lr, lr_multiplier, io_noise, perplexity,
                       attention_beam_width, headers):
    network = actrnn.ACTRNN(hidden_width=hidden_width, hidden_depth=hidden_depth,
                            io_width=io_width, io_noise=io_noise, perplexity=perplexity,
                            attention_beam_width=attention_beam_width).float()
    true_history = torch.zeros(max_history, io_width, requires_grad=False).float()
    pred_history = torch.zeros(max_history, io_width, requires_grad=False).float()
    act1_h = torch.zeros(hidden_depth, 1, hidden_width, requires_grad=True).float()
    state = {}
    state['init'] = {'true_history': true_history,
                     'predicted_history': pred_history,
                     'act1_h': act1_h}

    state['perplexity'] = perplexity
    state['max_history'] = max_history
    state['io_width'] = io_width
    state['attention_beam_width'] = attention_beam_width
    state['prime_lr'] = initial_lr
    state['lr_mul'] = lr_multiplier
    state['headers'] = headers
    state['prime_length'] = None
    state['norm_boundaries'] = None
    state['step_size'] = None
    return network, state

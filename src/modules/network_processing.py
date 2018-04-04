import copy
from math import isnan
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from src.GenerativeForecast import cuda
from src.modules.net_def import net
from . import data_proc


def create_forecasts(data_frame, network, state,  number_of_forecasts, future_length, noise_amount):
    if data_frame:
        criterion = nn.SmoothL1Loss().float()
        input = Variable(torch.from_numpy(data_frame['x']), requires_grad=False).float()
        target = Variable(torch.from_numpy(data_frame['y']), requires_grad=False).float()
        if cuda:
            input = input.cuda()
            target = target.cuda()
        output = network.forward(input=input)
        state = network.get_state()
        update_loss = criterion(output, target)
        update_loss = update_loss.cpu().data.numpy()[0]
        print("model update loss: {}".format(str(update_loss)))
    print("checkpoint state loaded, beginning to forecast")
    normal_forecasts = np.empty((number_of_forecasts, future_length, state['io_width']), dtype=np.float)
    raw_forecasts = np.empty((number_of_forecasts, future_length, state['io_width']), dtype=np.float)
    for i in range(number_of_forecasts):
        network.load_mutable_state(state)
        network.perturb(noise_amount=noise_amount)
        result = network.forecast(future=future_length).cpu().data.numpy()
        denormalized = data_proc.revert_normalization(result, state)
        normal_forecasts[i] = result
        raw_forecasts[i] = denormalized
        print("forecast {} complete".format(str(i)))
    network.load_mutable_state(state)
    return normal_forecasts, raw_forecasts, state




def train_autogenerative_model(data_frame, network, checkpoint_state, iterations):
    input = Variable(torch.from_numpy(data_frame['x']), requires_grad=False).float()
    target = Variable(torch.from_numpy(data_frame['y']), requires_grad=False).float()[0]
    if cuda:
        input = input.cuda()
        target = target.cuda()
    criterion = nn.MSELoss()
    best_loss = 1
    best_state = None
    optimizer = optim.Adagrad(network.parameters())
    print("ready to pre-train")
    cur_iter = 0
    while True:
        if cur_iter >= iterations:
            break
        diff_time = perf_counter()
        network.load_mutable_state(checkpoint_state)
        optimizer.zero_grad()
        network.reset_history()
        output = network(input)
        loss = criterion(output, target)
        loss_cpu = loss.detach().cpu().data.numpy()[0]

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
        # after_proc_t = perf_counter()
        loss.backward()
        # loss_t = perf_counter()
        # print("backprop took: {} s".format(str(loss_t - after_proc_t)))
        network.gradClamp()
        print('total time: {}'.format(str(perf_counter() - diff_time)))
        diff_time = perf_counter()
        optimizer.step()
        # optim_t = perf_counter()
        # print("optimization took: {} s".format(str(optim_t - diff_time)))
    network.load_mutable_state(checkpoint_state)
    network.forward(input)
    print('best overall training loss: {}'.format(str(best_loss)))
    return best_loss, network

# this determines the learning rate based on comparing the prime length to the current incremental length
def determine_lr(data, state):
    incr_length = len(data['x'])
    ratio = float(incr_length) / state['prime_length']
    incr_lr = ratio*state['prime_lr']
    return incr_lr



def initialize_network(io_dim, layer_width, max_history, initial_lr, lr_multiplier, io_noise, training_length,
                       attention_beam_width, headers):
    network = net.Net(hidden_width=layer_width, io_width=io_dim, max_history=max_history, initial_lr=initial_lr,
                      lr_multiplier=lr_multiplier, io_noise=io_noise, attention_beam_width=attention_beam_width,
                      headers=headers, training_length=training_length).float()
    if cuda:
        network = network.cuda()
    state = network.get_state()
    return network, state
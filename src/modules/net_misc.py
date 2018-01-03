from uuid import uuid4
from src.modules import graph, net, misc, data_proc
import numpy as np
import torch
import copy
from random import randrange
from torch import nn
from torch import optim
from torch.autograd import Variable
from math import isnan


def create_forecasts(data_frame, network, state,  number_of_forecasts, future_length, noise_amount):
    if data_frame:
        criterion = nn.SmoothL1Loss().cuda().float()
        input = Variable(torch.from_numpy(data_frame['x']), requires_grad=False).cuda().float()
        target = Variable(torch.from_numpy(data_frame['y']), requires_grad=False).cuda().float()
        output = network.forward(input=input)
        state = network.get_state()
        update_loss = criterion(output, target)
        update_loss = update_loss.cpu().data.numpy()[0]
        print("model update loss: {}".format(str(update_loss)))
    normal_forecasts = list()
    raw_forecasts = list()
    print("checkpoint state loaded, beginning to forecast")
    for i in range(number_of_forecasts):
        network.load_mutable_state(state)
        network.perturb(noise_amount=noise_amount)
        result = np.swapaxes(network.forecast(future=future_length).cpu().data.numpy(), 0, 1)
        # We only take the first element of the beam even if our model has a large beam.
        only_first_step = result[:, 0]
        denormalized = data_proc.revert_normalization(only_first_step, state)
        normal_forecasts.append(denormalized)
        raw_forecasts.append(only_first_step)
        print("forecast {} complete".format(str(i)))
    normal_forecasts = np.swapaxes(np.asarray(normal_forecasts), 0, 1)
    raw_forecasts = np.swapaxes(np.asarray(raw_forecasts), 0, 1)
    network.load_mutable_state(state)
    return normal_forecasts, raw_forecasts, state


def evaluate_performance(data, network, checkpoint_state, length):
    input = Variable(torch.from_numpy(data['x']), requires_grad=False).cuda().float()
    lower_forecast_bound = 0
    upper_forecast_bound = input.shape[0] - length

    print("ready to forcast_train")
    for _ in range(5):
        network.load_mutable_state(checkpoint_state)
        t_f = randrange(lower_forecast_bound, upper_forecast_bound)
        back_input = input[0:t_f]
        forecast_target = input[t_f:t_f + length]
        back_output = network.forward(back_input)
        forecast = network.forecast(future=length)
        graph.graph_training_data(forecast, forecast_target, back_output, back_input)

    # network.load_state_dict(best_state)
    network.load_mutable_state(checkpoint_state)
    network.forward(input)
    return network


def train_autogenerative_model(data_frame, network, checkpoint_state, learning_rate, epochs, iterations, drop_percentage):
    input = Variable(torch.from_numpy(data_frame['x']), requires_grad=False).cuda().float()
    target = Variable(torch.from_numpy(data_frame['y']), requires_grad=False).cuda().float()
    criterion = network.modied_mse_forecast_loss
    best_loss = 1
    best_state = None
    optimizer = optim.LBFGS(network.parameters(), lr=learning_rate, max_iter=iterations)
    print("ready to pre-train")
    for i in range(epochs):
        network.load_mutable_state(checkpoint_state)
        print("epoch: {}".format(str(i)))
        def closure():
            nonlocal best_loss
            nonlocal best_state
            nonlocal input
            nonlocal target
            nonlocal network
            optimizer.zero_grad()
            network.reset_history()
            output = network(input, drop=drop_percentage)
            loss = criterion(output, target)
            loss_cpu = loss.cpu().data.numpy()[0]

            if loss_cpu <= best_loss:
                best_loss = loss_cpu
                best_state = copy.deepcopy(network.state_dict())
                print('current best pre-training loss: {}'.format(str(best_loss)))
            elif loss_cpu > best_loss*10 or isnan(loss_cpu):
                print("loss was {}\nover threshold, reloading to best state".format(str(loss_cpu)))
                network.load_state_dict(best_state)
                optimizer.zero_grad()
            else:
                print('pre-training loss: {}'.format(str(loss_cpu)))
            loss.backward(retain_graph=True)
            net.gradClamp(network.parameters())
            return loss
        optimizer.step(closure)
    network.load_mutable_state(checkpoint_state)
    network.forward(input)
    print('best overall pre-training loss: {}'.format(str(best_loss)))
    return best_loss, network


# this determines the learning rate based on comparing the prime length to the current incremental length
def determine_lr(data, state):
    incr_length = len(data['x'])
    ratio = float(incr_length) / state['prime_length']
    incr_lr = ratio*state['prime_lr']
    return incr_lr


def load_checkpoint(network_path):
    network = load_model(network_path).cuda().float()
    state = network.get_state()
    return network, state


def initialize_network(io_dim, layer_width, max_history, initial_lr, lr_multiplier, io_noise,
                       attention_beam_width, future_beam_width, headers):
    network = net.Net(layer_width=layer_width, io_width=io_dim, max_history=max_history, initial_lr=initial_lr,
                      lr_multiplier=lr_multiplier, io_noise=io_noise, attention_beam_width=attention_beam_width,
                      future_beam_width=future_beam_width, headers=headers).cuda().float()
    state = network.get_state()
    return network, state


def save_model(checkpoint_file, remote_uri):
    local_file_path = "/tmp/{}.t7".format(str(uuid4()))
    torch.save(checkpoint_file, local_file_path)
    return misc.put_file(local_file_path, remote_uri)


def load_model(local_path):
    try:
        net = torch.load(local_path)
        return net
    except Exception as e:
        raise misc.AlgorithmError("unable to load model file: {}".format(e.args))
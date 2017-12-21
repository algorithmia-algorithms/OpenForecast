#!/usr/bin/env python
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import math
from random import randrange

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            rng = torch.autograd.Variable(torch.randn(din.size()) * self.stddev).cuda().float()
            return din + rng
        return din





class Net(nn.Module):
    def __init__(self, layer_width=45,
                 initial_lr=0.8,
                 lr_multiplier=1,
                 max_history=50,
                 io_width=1,
                 io_noise=0.04,
                 lookup_beam_width=1,
                 future_beam_width=1,
                 headers=None):
        super(Net, self).__init__()
        self.layer_width = layer_width
        self.max_history = max_history
        self.prime_lr = initial_lr
        self.lr_multiplier = lr_multiplier
        self.norm_boundaries = None
        self.prime_length = None
        self.io_width = io_width
        self.headers = headers
        self.io_noise = GaussianNoise(stddev=io_noise)
        self.depth = int(math.ceil(math.log(io_width))+1)
        self.lookup_beam_width = lookup_beam_width
        self.output_beam_width = future_beam_width
        self.lin1 = nn.Linear(io_width * self.lookup_beam_width, self.layer_width)
        self.gru1 = nn.GRU(self.layer_width, self.layer_width, self.depth)
        self.lin2 = nn.Linear(self.layer_width, self.io_width * self.output_beam_width)
        self.true_history = Variable(torch.zeros(self.max_history, self.io_width), requires_grad=False).cuda().float()
        self.pred_history = Variable(torch.zeros(self.max_history, self.io_width), requires_grad=False).cuda().float()
        self.gru1_h = Variable(torch.zeros(self.depth, 1, self.layer_width), requires_grad=False).cuda().float()

    def perturb(self, noise_amount=0):
        noise = GaussianNoise(stddev=noise_amount)
        self.gru1_h = noise(self.gru1_h)

    def load_mutable_state(self, state):
        self.gru1_h = state['gru1_h']
        self.pred_history = state['pred_history']
        self.true_history = state['true_history']

    def get_state(self):
        state = dict()
        state['depth'] = self.depth
        state['max_history'] = self.max_history
        state['true_history'] = self.true_history
        state['prime_length'] = self.prime_length
        state['io_width'] = self.io_width
        state['input_beam_width'] = self.lookup_beam_width
        state['output_beam_width'] = self.output_beam_width
        state['gru1_h'] = self.gru1_h
        state['pred_history'] = self.pred_history
        state['norm_boundaries'] = self.norm_boundaries
        state['prime_lr'] = self.prime_lr
        state['lr_mul'] = self.lr_multiplier
        state['headers'] = self.headers
        return state

    def initialize_meta(self, prime_length, norm_boundaries):
        self.norm_boundaries = norm_boundaries
        self.prime_length = prime_length

    def clear_state(self):
        self.gru1_h = Variable(torch.zeros(self.depth, 1, self.layer_width), requires_grad=False).cuda().float()
        self.reset_history()


    def reset_history(self):
        self.pred_history = Variable(torch.zeros(self.max_history, self.io_width),
                                     requires_grad=False).cuda().float()
        self.true_history = Variable(torch.zeros(self.max_history, self.io_width),
                                     requires_grad=False).cuda().float()

    def forecast(self, future=0):
        outputs = []
        for i in range(future):
            point = self.pred_history[-1]
            line = self.create_line(point)
            output = self.process(line)
            self.pred_history = torch.cat((self.pred_history, output[0]))
            self.pred_history = self.pred_history[1:]
            outputs.append(output)
        outputs = torch.cat(outputs, 1)
        return outputs


    def forward(self, input, drop=0.0):
        outputs = []
        rand_min = int(drop*1000)
        rand_max = int(drop*1000 + 1000)
        r = 1000
        for input_t in input:
            real_input = input_t.view(1, self.io_width)
            if r > 1000:
                point = self.pred_history[-1]
            else:
                point = real_input
            line = self.create_line(point)
            output = self.process(line)
            self.true_history = torch.cat((self.true_history, real_input))
            self.true_history = self.true_history[1:]
            self.pred_history = torch.cat((self.pred_history, output[0]))
            self.pred_history = self.pred_history[1:]
            outputs.append(output)
            r = randrange(rand_min, rand_max)
        outputs = torch.cat(outputs, 1)
        return outputs

    def create_line(self, point):
        line_data = list()
        line_data.append(point.squeeze())
        for i in range(2, self.lookup_beam_width+1):
            step = -i
            hist_point = self.pred_history[step]
            line_data.append(hist_point)
        line = torch.cat(line_data, 0).view(1, self.io_width * self.lookup_beam_width)
        return line


    def process(self, input):
        input = self.io_noise(input)
        lin_1 = self.lin1(input).view(1, 1, self.layer_width)
        gru_out, self.gru1_h = self.gru1(lin_1, self.gru1_h)
        gru_out = gru_out.view(1, self.layer_width)
        output = self.lin2(gru_out).view(self.output_beam_width, 1, self.io_width)
        return output

    def modied_mse_forecast_loss(self, input, target):
        mse_loss = nn.MSELoss().cuda()
        forecast_loss = Variable(torch.zeros(1)).cuda()
        for i in range(self.output_beam_width):
            input_slice = input[i]
            target_slice = target[i]
            slice_loss = mse_loss(input_slice, target_slice)
            slice_loss = torch.div(slice_loss, (i+1))
            forecast_loss = torch.add(forecast_loss, slice_loss)
        forecast_loss = torch.div(forecast_loss, self.output_beam_width)
        return forecast_loss

def gradClamp(parameters, clip=2):
    for p in parameters:
        p.grad.data.clamp_(max=clip)

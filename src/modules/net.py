#!/usr/bin/env python
import torch
from torch import nn
from torch.autograd import Variable
import math
import copy
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
    def __init__(self, state):
        super(Net, self).__init__()
        self.hidden_width = state['hidden_width']
        self.max_history = state['max_history']
        self.prime_lr = state['prime_lr']
        self.lr_multiplier = state['lr_multiplier']
        if 'norm_boundaries' in state:
            self.norm_boundaries = state['norm_boundaries']
        else:
            self.norm_boundaries = None
        self.prime_length = state['prime_length']
        self.io_width = state['io_width']
        self.headers = state['headers']
        self.stride_width = 2
        self.io_noise = GaussianNoise(stddev=state['io_noise'])
        # self.depth = int(math.log(self.prime_length)-4) + math.ceil(math.log(self.io_width))
        # if self.depth < 2:
        #     self.depth = 2
        # if self.depth > 8:
        #     self.depth = 7
        self.depth = 5
        self.attention_beam_width = state['attention_beam_width']
        self.output_beam_width = state['output_beam_width']
        self.lin1 = nn.Linear(self.io_width * self.attention_beam_width, self.hidden_width)
        # self.gru1 = drnn.DRNN(self.hidden_width, self.hidden_width, self.depth, stride_width=self.stride_width)
        self.gru1 = nn.GRU(self.hidden_width, self.hidden_width, self.depth)
        self.lin2 = nn.Linear(self.hidden_width, self.io_width * self.output_beam_width)
        self.true_history = Variable(torch.zeros(self.max_history, self.io_width), requires_grad=False).cuda().float()
        self.pred_history = Variable(torch.zeros(self.max_history, self.io_width), requires_grad=False).cuda().float()
        # self.gru1_h = Variable(torch.zeros(self.stride_width ** self.depth, 1, 1, self.hidden_width)).cuda().float()
        self.gru1_h = Variable(torch.zeros(self.depth, 1, self.hidden_width), requires_grad=False).cuda().float()

    def perturb(self, noise_amount=0):
        noise = GaussianNoise(stddev=noise_amount)
        self.gru1_h = [noise(vec) for vec in self.gru1_h]
        self.pred_history = noise(self.pred_history)

    def load_mutable_state(self, state):
        self.gru1_h = state['gru1_h']
        self.pred_history = state['pred_history']
        self.true_history = state['true_history']

    def copy_model(self):
        state = self.get_state()
        copied_dict = copy.deepcopy(self.state_dict())
        return state, copied_dict

    def get_state(self):
        state = dict()
        state['depth'] = self.depth
        state['max_history'] = self.max_history
        state['true_history'] = self.true_history
        state['pred_history'] = self.pred_history
        state['prime_length'] = self.prime_length
        state['io_width'] = self.io_width
        state['hidden_width'] = self.hidden_width
        state['attention_beam_width'] = self.attention_beam_width
        state['output_beam_width'] = self.output_beam_width
        state['gru1_h'] = self.gru1_h
        state['io_noise'] = self.io_noise
        state['norm_boundaries'] = self.norm_boundaries
        state['prime_lr'] = self.prime_lr
        state['lr_multiplier'] = self.lr_multiplier
        state['headers'] = self.headers
        return state

    def initialize_meta(self, prime_length, norm_boundaries):
        self.norm_boundaries = norm_boundaries
        self.prime_length = prime_length

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
            self.true_history = self.pred_history
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
        for i in range(2, self.attention_beam_width+1):
            step = -i
            hist_point = self.true_history[step]
            line_data.append(hist_point)
        line = torch.cat(line_data, 0).view(1, self.io_width * self.attention_beam_width)
        return line


    def process(self, input):
        input = self.io_noise(input)
        lin_1 = self.lin1(input).view(1, 1, self.hidden_width)
        gru_out, self.gru1_h = self.gru1.forward(lin_1, self.gru1_h)
        # gru_out, self.gru1_h = self.gru1.forward(lin_1, self.gru1_h)
        gru_out = gru_out.view(1, self.hidden_width)
        output = self.lin2(gru_out).view(self.output_beam_width, 1, self.io_width)
        return output

    def modied_mse_forecast_loss(self, input, target):
        mse_loss = nn.MSELoss().cuda()
        loss = mse_loss(input[0], target[0])
        return loss

def gradClamp(parameters, clip=2):
    for p in parameters:
        p.grad.data.clamp_(max=clip)

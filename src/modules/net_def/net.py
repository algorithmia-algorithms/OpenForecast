import torch
from torch import nn
from torch.autograd import Variable
from .act import ACTNN
import numpy as np
class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            rng = torch.autograd.Variable(torch.randn(din.size()) * self.stddev).float()
            return din + rng
        return din



class Net(nn.Module):
    def __init__(self, hidden_width=45,
                 initial_lr=0.8,
                 lr_multiplier=1,
                 max_history=50,
                 io_width=1,
                 perplexity=3,
                 io_noise=0.04,
                 attention_beam_width=1,
                 headers=None):
        super(Net, self).__init__()
        self.hidden_width = hidden_width
        self.max_history = max_history
        self.prime_lr = initial_lr
        self.lr_multiplier = lr_multiplier
        self.norm_boundaries = None
        self.prime_length = None
        self.step_size = None
        self.io_width = io_width
        self.headers = headers
        self.perplexity = perplexity
        self.io_noise = GaussianNoise(stddev=io_noise)
        self.act_hidden = self.hidden_width + 1
        self.attention_beam_width = attention_beam_width
        self.input_atten = nn.Linear(self.hidden_width, self.attention_beam_width*self.io_width)
        self.input_processing = nn.Linear(self.io_width * self.attention_beam_width, self.hidden_width)
        self.act1 = ACTNN(input_size=self.hidden_width, hidden_size=self.hidden_width, output_size=self.io_width, M=4, perplexity=self.perplexity)
        self.true_history = Variable(torch.zeros(self.max_history, self.io_width), requires_grad=False).float()
        self.pred_history = Variable(torch.zeros(self.max_history, self.io_width), requires_grad=False).float()
        self.act1_h = Variable(torch.zeros(1, 1, self.hidden_width)).float()
        self.ponder_cost = 0
        self.ponder_times = []

    def perturb(self, noise_amount=0):
        noise = GaussianNoise(stddev=noise_amount)
        self.act1_h = noise(self.act1_h)
        self.pred_history = noise(self.pred_history)

    def load_mutable_state(self, state):
        self.act1_h = state['act1_h']
        self.pred_history = state['pred_history']
        self.true_history = state['true_history']

    def get_state(self):
        state = dict()
        state['perplexity'] = self.perplexity
        state['max_history'] = self.max_history
        state['true_history'] = self.true_history
        state['pred_history'] = self.pred_history
        state['prime_length'] = self.prime_length
        state['io_width'] = self.io_width
        state['attention_beam_width'] = self.attention_beam_width
        state['act1_h'] = self.act1_h
        state['norm_boundaries'] = self.norm_boundaries
        state['prime_lr'] = self.prime_lr
        state['lr_mul'] = self.lr_multiplier
        state['headers'] = self.headers
        state['step_size'] = self.step_size
        return state

    def initialize_meta(self, prime_length, norm_boundaries, step_size):
        self.norm_boundaries = norm_boundaries
        self.prime_length = prime_length
        self.step_size = step_size

    def zero(self):
        self.reset_history()
        self.act1_h = Variable(torch.zeros(1, 1, self.hidden_width)).float()

    def reset_history(self):
        self.pred_history = Variable(torch.zeros(self.max_history, self.io_width),
                                     requires_grad=False).float()
        self.true_history = Variable(torch.zeros(self.max_history, self.io_width),
                                     requires_grad=False).float()

    def forecast(self, future=0):
        outputs = Variable(torch.zeros(future, self.io_width), requires_grad=False).float()
        ponder_costs = 0
        for i in range(future):
            input_t = self.pred_history[-1].view(1, self.io_width)
            line = self.create_line(input_t)
            output, ponder_cost, num_steps = self.process(line)
            ponder_costs += ponder_cost
            self.pred_history = torch.cat((self.pred_history, output))
            self.pred_history = self.pred_history[1:]
            self.true_history = self.pred_history
            outputs[i] = output
        return outputs


    def forward(self, input):
        length_of_input = input.shape[0]
        outputs = Variable(torch.zeros(length_of_input, self.io_width), requires_grad=False).float()
        ponder_costs = 0
        total_steps = []
        for i in range(length_of_input):
            input_t = input[i].view(1, self.io_width)
            line = self.create_line(input_t)
            output, ponder_cost, num_steps = self.process(line)
            total_steps.append(num_steps)
            self.state_history = torch.cat((self.state_history, self.act1_h))
            self.state_history = self.state_history[1:, :]
            self.true_history = torch.cat((self.true_history, input_t))
            self.true_history = self.true_history[1:, :]
            self.pred_history = torch.cat((self.pred_history, output))
            self.pred_history = self.pred_history[1:, :]
            ponder_costs += ponder_cost
            outputs[i] = output
            ponder_costs = ponder_costs / length_of_input
        test = np.asarray(total_steps[-1000:])
        timing_data = {}
        for time in total_steps:
            if str(time) in timing_data.keys():
                timing_data[str(time)] += 1
            else:
                timing_data[str(time)] = 1
        print("number of iterations:")
        for key in timing_data.keys():
            print('{}: {}'.format(key, str(timing_data[key])))
        return outputs, ponder_costs

    def create_line(self, point):
        # hist_point = self.true_history[-self.attention_beam_width:-1]
        hist_point = self.pred_history[-self.attention_beam_width:-1]
        line = torch.cat((hist_point, point), 0)
        line = line.view(1, self.io_width * self.attention_beam_width)
        return line


    def process(self, input):
        input = self.io_noise(input)
        lin_1 = self.input_processing(input)
        output, self.act1_h, ponder_cost, num_steps = self.act1(lin_1, self.act1_h)
        return output, ponder_cost, num_steps

    def modied_mse_forecast_loss(self, input, target):
        mse_loss = nn.MSELoss()
        loss = mse_loss(input, target)
        return loss

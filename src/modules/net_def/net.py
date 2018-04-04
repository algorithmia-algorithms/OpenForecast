import torch
from torch import nn
from torch.autograd import Variable
import math
cuda = False

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            rng = torch.autograd.Variable(torch.randn(din.size()) * self.stddev).float()
            if cuda:
                rng = rng.cuda()
            return din + rng
        return din



class Net(nn.Module):
    def __init__(self, hidden_width=45,
                 initial_lr=0.8,
                 lr_multiplier=1,
                 max_history=50,
                 io_width=1,
                 training_length=100,
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
        self.io_width = io_width
        self.headers = headers
        self.io_noise = GaussianNoise(stddev=io_noise)
        self.depth = int(math.log(training_length)-4) + math.ceil(math.log(self.io_width))
        if self.depth < 2:
            self.depth = 2
        if self.depth > 8:
            self.depth = 7
        self.attention_beam_width = attention_beam_width
        self.lin1 = nn.Linear(io_width * self.attention_beam_width, self.hidden_width)
        self.gru1 = nn.GRU(self.hidden_width, self.hidden_width, self.depth)
        self.lin2 = nn.Linear(self.hidden_width, self.io_width)
        self.true_history = Variable(torch.zeros(self.max_history, self.io_width), requires_grad=False).float()
        self.pred_history = Variable(torch.zeros(self.max_history, self.io_width), requires_grad=False).float()
        self.gru1_h = Variable(torch.zeros(self.depth, 1, self.hidden_width)).float()
        if cuda:
            self.true_history = self.true_history.cuda()
            self.pred_history = self.pred_history.cuda()
            self.gru1_h = self.gru1_h.cuda()

    def perturb(self, noise_amount=0):
        noise = GaussianNoise(stddev=noise_amount)
        self.gru1_h = noise(self.gru1_h)
        self.pred_history = noise(self.pred_history)

    def load_mutable_state(self, state):
        self.gru1_h = state['gru1_h']
        self.pred_history = state['pred_history']
        self.true_history = state['true_history']

    def get_state(self):
        state = dict()
        state['depth'] = self.depth
        state['max_history'] = self.max_history
        state['true_history'] = self.true_history
        state['pred_history'] = self.pred_history
        state['prime_length'] = self.prime_length
        state['io_width'] = self.io_width
        state['attention_beam_width'] = self.attention_beam_width
        state['gru1_h'] = self.gru1_h
        state['norm_boundaries'] = self.norm_boundaries
        state['prime_lr'] = self.prime_lr
        state['lr_mul'] = self.lr_multiplier
        state['headers'] = self.headers
        return state

    def initialize_meta(self, prime_length, norm_boundaries):
        self.norm_boundaries = norm_boundaries
        self.prime_length = prime_length

    def reset_history(self):
        self.pred_history = Variable(torch.zeros(self.max_history, self.io_width),
                                     requires_grad=False).float()
        self.true_history = Variable(torch.zeros(self.max_history, self.io_width),
                                     requires_grad=False).float()
        if cuda:
            self.pred_history = self.pred_history.cuda()
            self.true_history = self.true_history.cuda()

    def forecast(self, future=0):
        outputs = Variable(torch.zeros(future, self.io_width), requires_grad=False).float()
        for i in range(future):
            input_t = self.pred_history[-1].view(1, self.io_width)
            line = self.create_line(input_t)
            output = self.process(line)
            self.pred_history = torch.cat((self.pred_history, output))
            self.pred_history = self.pred_history[1:]
            self.true_history = self.pred_history
            outputs[i] = output
        return outputs


    def forward(self, input):
        length_of_input = input.shape[0]
        outputs = Variable(torch.zeros(length_of_input, self.io_width), requires_grad=False).float()
        if cuda:
            outputs = outputs.cuda()
        for i in range(length_of_input):
            input_t = input[i].view(1, self.io_width)
            line = self.create_line(input_t)
            output = self.process(line)
            self.true_history = torch.cat((self.true_history, input_t))
            self.true_history = self.true_history[1:, :]
            self.pred_history = torch.cat((self.pred_history, output))
            self.pred_history = self.pred_history[1:, :]
            outputs[i] = output
        return outputs

    def create_line(self, point):
        # for i in range(2, self.attention_beam_width+1):
        hist_point = self.true_history[-self.attention_beam_width:-1]
        line = torch.cat(((hist_point, point)), 0)
        line = line.view(1, self.io_width * self.attention_beam_width)
        return line


    def process(self, input):
        input = self.io_noise(input)
        lin_1 = self.lin1(input).view(1, 1, self.hidden_width)
        gru_out, self.gru1_h = self.gru1.forward(lin_1, self.gru1_h)
        gru_out = gru_out.view(1, self.hidden_width)
        output = self.lin2(gru_out).view(1, self.io_width)
        return output

    def modied_mse_forecast_loss(self, input, target):
        mse_loss = nn.MSELoss()
        loss = mse_loss(input, target)
        return loss

    def gradClamp(self, clip=2):
        for p in self.parameters():
            p.grad.data.clamp_(max=clip)

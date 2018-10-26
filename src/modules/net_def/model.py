import torch
from torch import nn
from .dialatedRNN import DialatedRNN

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            rng = torch.autograd.Variable(torch.randn(din.size()) * self.stddev).float()
            return din + rng
        return din



class Model(nn.Module):

    __constants__ = ['io_width', 'input_width', 'linear_width', 'memory_width']

    def __init__(self, memory_width=45,
                 io_width=1,
                 io_noise=0.04,
                 linear_width=10,
                 depth=2,
                 input_width=5):
        super(Model, self).__init__()

        self.io_width = io_width
        self.input_width = input_width
        self.linear_width = linear_width
        self.memory_width = memory_width
        self.io_noise = torch.jit.trace(GaussianNoise(stddev=io_noise), example_inputs=(torch.randn(io_width*5)), check_trace=False)
        self.proc = torch.jit.trace(nn.Linear(self.io_width * self.input_width, self.linear_width), example_inputs=(torch.randn(self.io_width * 5)))
        self.actrnn = DialatedRNN(input_size=linear_width, working_size=self.memory_width, depth=depth, output_size=self.io_width)




    def forecast(self, future_length: int, historical_data: torch.Tensor, residual: torch.Tensor, memory: torch.Tensor, init_step: int):
        io_width = self.io_width
        for i in range(future_length):
            step = i + init_step
            x_t = historical_data[-5:]
            output, residual, memory = self.process(x_t, residual, memory, step)
            historical_data = torch.cat((historical_data, output.view(1, io_width)))
            historical_data = historical_data[1:]
        outputs = historical_data[-future_length:]
        return outputs




    def forward(self, input: torch.Tensor, true_history: torch.Tensor,
                residual: torch.Tensor, memory: torch.Tensor, input_length: int, init_step: int):
        for i in range(input_length):
            step = i + init_step
            true_history = torch.cat((true_history, input[i].view(1, self.io_width)))
            true_history = true_history[1:]
            x_t = true_history[-5:]
            _, residual, memory = self.process(x_t, residual, memory, step)
        return true_history, residual, memory


    def process(self, input: torch.Tensor, residual: torch.Tensor, hidden: torch.Tensor, step: int):
        input = input.view(self.io_width*self.input_width)
        input = self.io_noise(input)
        ready = self.proc(input)
        output, residual, hidden = self.actrnn(ready, residual, hidden, step)
        output = output
        return output, residual, hidden

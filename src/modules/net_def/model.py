import torch
from torch import nn

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

    __constants__ = ['io_width', 'linear_width', 'memory_width']

    def __init__(self, memory_width=45,
                 io_width=1,
                 io_noise=0.04,
                 linear_width=10,
                 depth=2):
        super(Model, self).__init__()

        self.io_width = io_width
        self.linear_width = linear_width
        self.memory_width = memory_width
        # self.io_noise = torch.jit.trace(GaussianNoise(stddev=io_noise), example_inputs=(torch.randn(io_width)), check_trace=False)
        # self.proc = torch.jit.trace(nn.Linear(self.io_width, self.linear_width), example_inputs=(torch.randn(self.io_width)))
        # self.rnn = torch.jit.trace(nn.GRU(input_size=self.linear_width, hidden_size=self.memory_width, num_layers=depth),
        #                            example_inputs=(torch.randn(1, 1, self.linear_width), torch.randn(depth, 1, self.memory_width)))
        # self.output = torch.jit.trace(nn.Linear(self.memory_width, self.io_width), example_inputs=(torch.randn(self.memory_width)))

        self.io_noise = GaussianNoise(stddev=io_noise)
        self.proc = nn.Linear(self.io_width, self.linear_width)
        self.rnn = nn.GRU(input_size=self.linear_width, hidden_size=self.memory_width, num_layers=depth)
        self.output = nn.Linear(self.memory_width, self.io_width)

    def forecast(self, future_length: int, residual: torch.Tensor, memory: torch.Tensor,
                 starting_point: torch.Tensor, forecast_tensor: torch.Tensor):
        x_t = starting_point
        for i in range(future_length):
            output, residual, memory = self.process(x_t, residual, memory)
            forecast_tensor[i] = output
            x_t = forecast_tensor[i]
        return forecast_tensor


    def forward(self, input: torch.Tensor, residual: torch.Tensor, memory: torch.Tensor, input_length: int):
        for i in range(input_length-1):
            x_t = input[i]
            x_t = self.io_noise(x_t)
            _, residual, memory = self.process(x_t, residual, memory)
        return residual, memory


    def process(self, input: torch.Tensor, residual: torch.Tensor, memory: torch.Tensor):
        input = input.view(self.io_width)
        processed_input = self.proc(input)
        processed_input = processed_input.view(1, 1, -1)
        rnn_out, memory = self.rnn(processed_input, memory)
        rnn_merged = rnn_out + residual
        residual = self.update_residual(rnn_out, residual)
        output = self.output(rnn_merged)
        output = output.squeeze()

        return output, residual, memory


    def update_residual(self, output, residual):
        residual += output
        residual = torch.sigmoid(residual)
        return residual

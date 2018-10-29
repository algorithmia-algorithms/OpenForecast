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
                 forecast_tensor: torch.Tensor):
        x_t = forecast_tensor[0]
        for i in range(future_length-1):
            output, residual, memory = self.forward(x_t, residual, memory)
            output = output.view(1, -1)
            forecast_tensor = torch.cat((forecast_tensor, output))
            x_t = output
        return forecast_tensor


    def forward(self, input: torch.Tensor, residual: torch.Tensor, memory: torch.Tensor):
        input = input.view(self.io_width)
        input = self.io_noise(input)
        processed_input = self.proc(input)
        processed_input = processed_input.view(1, 1, -1)
        rnn_out, memory = self.rnn(processed_input, memory)
        rnn_merged = rnn_out + residual
        residual = self.update_residual(rnn_out, residual)
        output = self.output(rnn_merged)

        return output, residual, memory


    def update_residual(self, output, residual):
        residual = residual + output
        residual = torch.sigmoid(residual)
        return residual

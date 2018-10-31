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


class ForecastModel(torch.jit.ScriptModule):


    def __init__(self, architecture):
        super(ForecastModel, self).__init__()

        linear_in_input = architecture['linear_in']['input']
        linear_in_output = architecture['linear_in']['output']
        linear_out_input = architecture['linear_out']['input']
        linear_out_output = architecture['linear_out']['output']
        memory_input = architecture['recurrent']['input']
        memory_output = architecture['recurrent']['output']
        memory_depth = architecture['recurrent']['depth']
        stddev = architecture['gaussian_noise']['stddev']
        io_noise = GaussianNoise(stddev)
        linear_in = nn.Linear(linear_in_input, linear_in_output)
        recurrent = nn.GRU(memory_input, memory_output, memory_depth)
        linear_out = nn.Linear(linear_out_input, linear_out_output)

        self.io_noise = torch.jit.trace(io_noise, example_inputs=(torch.randn(linear_in_input)), check_trace=False)
        self.linear_in = torch.jit.trace(linear_in, example_inputs=(torch.randn(linear_in_input)))
        self.recurrent = torch.jit.trace(recurrent, example_inputs=(torch.randn(1, 1, memory_input), torch.randn(memory_depth, 1, memory_output)))
        self.linear_out = torch.jit.trace(linear_out, example_inputs=(torch.randn(1, 1, linear_out_input)))

    @torch.jit.script_method
    def forward(self, input: torch.Tensor, residual: torch.Tensor, memory: torch.Tensor):
        input = input.view(-1)
        input = self.io_noise(input)
        processed_input = self.linear_in(input)
        processed_input = processed_input.view(1, 1, -1)
        rnn_out, memory = self.recurrent(processed_input, memory)
        rnn_merged = rnn_out + residual
        residual = self.update_residual(rnn_out, residual)
        output = self.linear_out(rnn_merged)

        return output, residual, memory

    @torch.jit.script_method
    def update_residual(self, output, residual):
        residual = residual + output
        residual = torch.sigmoid(residual)
        return residual

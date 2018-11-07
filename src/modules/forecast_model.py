import torch
from torch import nn

class GaussianNoise(torch.jit.ScriptModule):
    def __init__(self):
        super(GaussianNoise, self).__init__()

    def forward(self, din, stddev):
        rng = torch.autograd.Variable(torch.randn(din.size()) * stddev).float()
        return din + rng


class ForecastNetwork(torch.jit.ScriptModule):

    r"""
    This class defines the torch neural network architecture.
    The architecture itself is defined as follows:

    Residual(t-1)      X(t-1)      Memory(t-1)
    +                   +              +
    |                   |              |
    |                   |              |
    |            +------v-----+        |
    |            |Linear Layer|        |
    |            +------+-----+        |
    |                   |              |
    |            +------+-----+        |
    |            |Recurrent   <--------+
    |            |Module      +-------->
    |            +------+-----+        |
    |                   |              |
    |      +------------+------------+ |
    +------>  Residual  Connection   | |
    <------+  g(x) = sig(f(x) + x)   | |
    |      +------------+-----+------+ |
    |                   |              |
    |            +------+-----+        |
    |            |Linear Layer|        |
    |            +------+-----+        |
    |                   |              |
    |                   |              |
    v                   v              v
    Residual(t)        X'(t)       Memory(t)


    Where the shape of each layer and module is defined in the `data_utilities.define_network_geometry()` function.
    """

    def __init__(self, architecture):
        super(ForecastNetwork, self).__init__()
        linear_in_input_shape = architecture['linear_in']['input']
        linear_in_output_shape = architecture['linear_in']['output']
        linear_out_input_shape = architecture['linear_out']['input']
        linear_out_output_shape = architecture['linear_out']['output']
        memory_input_shape = architecture['recurrent']['input']
        memory_output_shape = architecture['recurrent']['output']
        memory_depth_shape = architecture['recurrent']['depth']
        stddev = architecture['gaussian_noise']['stddev']
        io_noise = GaussianNoise()
        linear_in = nn.Linear(linear_in_input_shape, linear_in_output_shape)
        recurrent = nn.GRU(memory_input_shape, memory_output_shape, memory_depth_shape)
        linear_out = nn.Linear(linear_out_input_shape, linear_out_output_shape)

        self.io_noise = torch.jit.trace(io_noise, example_inputs=(torch.randn(linear_in_input_shape), torch.tensor(stddev)), check_trace=False)
        self.linear_in = torch.jit.trace(linear_in, example_inputs=(torch.randn(linear_in_input_shape)))
        self.recurrent = torch.jit.trace(recurrent, example_inputs=(torch.randn(1, 1, memory_input_shape), torch.randn(memory_depth_shape, 1, memory_output_shape)))
        self.linear_out = torch.jit.trace(linear_out, example_inputs=(torch.randn(1, 1, linear_out_input_shape)))

    @torch.jit.script_method
    def forward(self, input: torch.Tensor, residual: torch.Tensor, memory: torch.Tensor, noise_stddev: torch.Tensor):
        input = input.view(-1)
        input = self.io_noise(input, noise_stddev)
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



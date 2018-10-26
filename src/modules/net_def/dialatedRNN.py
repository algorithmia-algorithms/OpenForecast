import torch
from torch import nn

r""" 
"""



class DialatedRNN(nn.Module):

    def __init__(self, input_size, working_size, output_size, depth):
        super(DialatedRNN, self).__init__()

        self.depth = depth

        self.rnn = []

        for i in range(depth):
            self.rnn.append(nn.GRU(input_size, working_size,
                           num_layers=1, batch_first=True))

        self.rnn = torch.nn.ModuleList(self.rnn)

        self.output_processor = nn.Linear(working_size, output_size)



    def forward(self, input: torch.Tensor, residual: list, memory: list, step: int):
        x = input.view(1, 1, -1)
        results = []
        residual_composition = 0
        positional_composition = 0
        for i in range(self.depth):
            itr = (i+1)**2
            if bool(step % itr == 0):
                x_t, residual[i], memory[i] = self.process_dialation(x, self.rnn[i], memory[i], residual[i])
                results.append(x_t)
        for i in range(self.depth):
            residual_composition += residual[i]/((i+1)**2)
        for i in range(len(results)):
            positional_composition += results[i]/((i+1)**2)
        composite = positional_composition + residual_composition
        output = self.output_processor(composite)

        return (output, residual, memory)



    def process_dialation(self, input, func, memory_tensor: torch.Tensor, residual_tensor: torch.Tensor):
        x_t, memory_tensor = func(input, memory_tensor)
        residual_tensor = torch.sigmoid(residual_tensor + x_t.view(-1))
        return (x_t, residual_tensor, memory_tensor)

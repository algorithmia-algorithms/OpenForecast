import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from time import perf_counter

class ACTNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, perplexity=0.99, M=100):
        super(ACTNN, self).__init__()
        self.eps = perplexity
        self.M = M
        self.rnn = nn.RNN(1 + input_size, hidden_size,
                          num_layers=num_layers, batch_first=True)  # Add one to the input size for the binary flag # Set initial bias to avoid problems at the beginning
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, state: tuple):
        states = []
        halt_prob = []
        input = input.view(-1)
        x0 = torch.cat((Variable(torch.Tensor([0])).float(), input))

        # First iteration
        n = 0
        halt_sum = torch.zeros([1]).float()
        # start = perf_counter()
        while halt_sum < self.eps and n < self.M:
            hidden = self.rnn(x0.view(1, 1, -1), state.view(1, 1, -1))[1]
            states.append(hidden.view(-1))
            halt_prob.append(torch.Tensor([F.sigmoid(states[n].sum()) + 0.05*n]))
            halt_sum = halt_sum + halt_prob[n]
            n += 1

        if len(halt_prob) > 1:
            residual = torch.Tensor([1 - torch.sum(torch.cat(halt_prob[:-1]))])  # Residual
        else:
            residual = torch.Tensor([1])
        # stop = perf_counter() - start
        halt_prob[n-1] = residual
        states_tensor = torch.stack(states, dim=1)
        halt_prob_tensor = torch.cat(halt_prob)
        hidden = torch.mv(states_tensor, halt_prob_tensor)
        output = self.fc_output(hidden).view(1, self.fc_output.out_features)
        hidden = hidden.view(1, 1, self.rnn.hidden_size)
        ponder = n + residual
        return output, hidden, ponder, n
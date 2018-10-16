import torch
from torch import nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    def __init__(self, stddev: float):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            rng = torch.autograd.Variable(torch.randn(din.size()) * self.stddev).float()
            din = rng + din
        return din

class ACTNN(torch.jit.ScriptModule):

    __constants__ = ['hidden_depth', 'perplexity', 'max_steps', 'loop_penalty']

    def __init__(self, input_size, hidden_width, output_size, hidden_depth, perplexity=0.99,
                 max_steps=5, loop_penalty=0.05, halt_noise=0.05):
        super(ACTNN, self).__init__()
        self.perplexity = perplexity
        self.max_steps = max_steps
        self.loop_penalty = loop_penalty
        self.hidden_depth = hidden_depth
        # self.rnn = torch.jit.trace(nn.RNN(input_size + 1, hidden_width,
        #                                   num_layers=self.hidden_depth, batch_first=True), example_inputs=(torch.rand(1, 1, input_size + 1), torch.rand(self.hidden_depth, 1, hidden_width)))
        self.rnn = torch.jit.trace(nn.RNN(input_size, hidden_width,
                                          num_layers=self.hidden_depth, batch_first=True), example_inputs=(torch.rand(1, 1, input_size), torch.rand(self.hidden_depth, 1, hidden_width)))
        self.proc = torch.jit.trace(nn.Linear(hidden_width, output_size), example_inputs=torch.rand(1, hidden_width))
        self.halt_noise = torch.jit.trace(GaussianNoise(stddev=halt_noise), example_inputs=torch.rand(1), check_trace=False)


    @torch.jit.script_method
    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        hidden = hidden.view(self.hidden_depth, 1, -1)
        x = input.view(1, 1, -1)
        _, hidden = self.rnn(x, hidden)
        hidden = hidden.view(1, -1)
        output = self.proc(hidden)
        return (output, hidden)


    # @torch.jit.script_method
    # def forward(self, input: torch.Tensor, hidden: torch.Tensor):
    #     states = []
    #     hiddens = []
    #     halt_probs = []
    #     input = input.view(-1)
    #     n = 0
    #     halt_sum = torch.zeros([1]).float()
    #     x0 = torch.cat([input, torch.zeros([1])])
    #     x0 = x0.view(1, 1, -1)
    #     hidden = self.rnn(x0, hidden.view(self.hidden_depth, 1, -1))[1]
    #     state = self.proc(hidden)
    #     # teacher_proc = self.teacher(hidden).view(-1)
    #     hiddens.append(hidden.view(-1))
    #     states.append(state.view(-1))
    #     loop_penalty = self.loop_penalty * n
    #     halt_probability = torch.Tensor([F.sigmoid(hiddens[n].sum()) + loop_penalty])
    #     halt_probability = self.halt_noise(halt_probability)
    #     halt_probs.append(halt_probability)
    #     halt_sum = halt_sum + halt_probs[n]
    #     n += 1
    #     x1 = torch.cat([input, torch.Tensor([1])])
    #     while halt_sum < self.perplexity and n < self.max_steps:
    #         hidden = self.rnn(x1.view(1, 1, -1), hidden.view(self.hidden_depth, 1, -1))[1]
    #         state = self.proc(hidden)
    #         # teacher_proc = self.teacher(hidden).view(-1)
    #         # teacher_vals.append(F.sigmoid(teacher_proc))
    #         hiddens.append(hidden.view(-1))
    #         states.append(state.view(-1))
    #         loop_penalty = self.loop_penalty*n
    #         halt_probability = torch.Tensor([F.sigmoid(hiddens[n].sum()) + loop_penalty])
    #         halt_probability = self.halt_noise(halt_probability)
    #         halt_probs.append(halt_probability)
    #         halt_sum = halt_sum + halt_probs[n]
    #         n += 1
    #
    #     if n > 1:
    #         ponder_res = torch.Tensor([1 - torch.sum(torch.cat(halt_probs[:-1]))])
    #     else:
    #         ponder_res = torch.Tensor([1 - torch.sum(torch.cat(halt_probs))])
    #
    #     reduced = halt_probs[0] + ponder_res
    #     states_tensor = torch.stack(states, dim=1)
    #     hiddens_tensor = torch.stack(hiddens, dim=1)
    #     halt_prob_tensor = torch.cat(reduced)
    #     output = torch.mv(states_tensor, halt_prob_tensor)
    #     hidden = torch.mv(hiddens_tensor, halt_prob_tensor)
    #     ponder = n + ponder_res
    #     return output, hidden, ponder, n
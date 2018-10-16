import torch
from torch import nn
import torch.nn.functional as F
from .act import ACTNN

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            rng = torch.autograd.Variable(torch.randn(din.size()) * self.stddev).float()
            return din + rng
        return din



class ACTRNN(torch.jit.ScriptModule):

    __constants__ = ['io_width', 'attention_beam_width']

    def __init__(self, hidden_width=45,
                 io_width=1,
                 perplexity=3,
                 io_noise=0.04,
                 attention_beam_width=1,
                 hidden_depth = 2):
        super(ACTRNN, self).__init__()

        self.io_width = io_width
        self.attention_beam_width = attention_beam_width
        self.io_noise = torch.jit.trace(GaussianNoise(stddev=io_noise),example_inputs=torch.rand(io_width), check_trace=False)
        self.input_proc = torch.jit.trace(nn.Linear(io_width, hidden_width), example_inputs=torch.rand(io_width))
        self.output = torch.jit.trace(nn.Linear(attention_beam_width, io_width), example_inputs=torch.rand(attention_beam_width))
        self.attention = torch.jit.trace(nn.Linear(hidden_width, attention_beam_width), example_inputs=torch.rand(hidden_width))
        self.actrnn = ACTNN(input_size=hidden_width, hidden_width=hidden_width,
                            output_size=attention_beam_width, hidden_depth=hidden_depth, max_steps=4,
                            perplexity=perplexity, halt_noise=0.00)



    # @torch.jit.script_method
    # def perturb(self, act1_h, noise_amount):
    #     noise = GaussianNoise(stddev=noise_amount)
    #     act1_h = noise(act1_h)
    #     return act1_h


    # @torch.jit.script_method
    # def get_state(self):
    #     state = dict()
    #     state['perplexity'] = self.perplexity
    #     state['max_history'] = self.max_history
    #     state['prime_length'] = self.prime_length
    #     state['io_width'] = self.io_width
    #     state['attention_beam_width'] = self.attention_beam_width
    #     state['act1_h'] = self.act1_h
    #     state['norm_boundaries'] = self.norm_boundaries
    #     state['prime_lr'] = self.prime_lr
    #     state['lr_mul'] = self.lr_multiplier
    #     state['headers'] = self.headers
    #     state['step_size'] = self.step_size
    #     return state

    # @torch.jit.script_method
    # def initialize(self, prime_length, norm_boundaries, step_size):
    #     self.norm_boundaries = norm_boundaries
    #     self.prime_length = prime_length
    #     self.step_size = step_size


    @torch.jit.script_method
    def forecast(self, future_length: int, pred_history: torch.Tensor, hidden: torch.Tensor):
        # ponder_costs = 0
        io_width = self.io_width
        for i in range(future_length):
            x_t, taught = self.create_input_feature(pred_history[-1], pred_history[-1])
            output, hidden, _, _ = self.process(x_t, hidden, io_width)
            # ponder_costs += ponder_cost
            pred_history = torch.cat((pred_history, output.view(1, io_width)))
            pred_history = pred_history[1:]
        outputs = pred_history[:, -future_length:]
        return outputs



    # @torch.jit.script_method
    # def init_state(self):
    #     true_history = torch.zeros(self.max_history, self.io_width).float()
    #     pred_history = torch.zeros(self.max_history, self.io_width).float()
    #     hidden = torch.zeros(self.layer_width, 1, self.hidden_width)
    #     return true_history, pred_history, hidden


    @torch.jit.script_method
    def forward(self, input: torch.Tensor, length_of_input: int,
                true_history: torch.Tensor,
                predicted_history: torch.Tensor, hidden: torch.Tensor):
        ponder_costs = torch.zeros(1)
        io_width = self.io_width
        for i in range(length_of_input):
            x_t, taught = self.create_input_feature(input[i], predicted_history[-1])
            output, hidden, ponder_cost, num_steps = self.process(x_t, hidden, io_width)
            true_history = torch.cat((true_history, input[i].view(1, io_width)))
            true_history = true_history[1:]
            predicted_history = torch.cat((predicted_history, output.view(1, io_width)))
            predicted_history = predicted_history[1:]
            ponder_costs += ponder_cost
        ponder_costs = ponder_costs / length_of_input - 0.99
        return predicted_history, true_history, hidden, ponder_costs

    @torch.jit.script_method
    def create_input_feature(self, truth, pred):
        teacher_selector = torch.randperm(11)[0]
        if bool(teacher_selector > 7):
            x_t = truth
            train_selected = 1
        else:
            x_t = pred
            train_selected = 0
        return x_t, train_selected

    @torch.jit.script_method
    def process(self, input: torch.Tensor, hidden: torch.Tensor, io_width: int):
        input = self.io_noise(input)
        lin_1_out = self.input_proc(input)
        # act_out, hidden, ponder_cost, num_steps = self.actrnn(lin_1_out, hidden)
        act_out, hidden = self.actrnn(lin_1_out, hidden)
        num_steps=4
        ponder_cost = torch.zeros(1)
        attention_vector = F.softmax(act_out, dim=0)
        hidden_mask = hidden[0][0:self.attention_beam_width]
        atten_mask = attention_vector * hidden_mask
        output = self.output(atten_mask).view(io_width)
        return output, hidden, ponder_cost, num_steps

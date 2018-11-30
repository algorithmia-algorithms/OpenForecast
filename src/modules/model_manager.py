from time import perf_counter

import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import from_numpy
from src.modules.forecast_model import ForecastNetwork

class GaussianNoise:
    def __init__(self, stddev: float):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def add_noise(self, din):
        rng = torch.autograd.Variable(torch.randn(din.size()) * self.stddev).float()
        return din + rng




class Model:


    def __init__(self, meta_data, network=None):
        self.residual_shape = meta_data['tensor_shape']['residual']
        self.memory_shape = meta_data['tensor_shape']['memory']
        self.data_dimensionality = meta_data['io_dimension']
        self.key_variables = meta_data['key_variables']
        self.forecast_length = meta_data['forecast_length']
        self.training_time = meta_data['training_time']
        self.noise = GaussianNoise(meta_data['io_noise'])
        if network:
            self.network = network
        else:
            self.network = init_network(meta_data['architecture'])

    def forecast(self, data: np.ndarray):
        r"""
        By comparison with training, the forecast process is might simpler.

        The forecast process is described as below:

                                                                                                           Output Forecast
                                                                                                           Y(n:n+l)
                                                                                                               ^
                                                                                                               |
                                                                                                               |
                                                                                                               |
 +---------+------+                 +---------+                                     +----------+               |
 |Data Set | X(t) +-------------+   |         |                                     |          |               |
 +---------+------+             +--->         |                                     |          |               |
                                    |         |                                     |  when    |          +----+----+
                                    |For each |                                     |  t = n   |  X'(n)   |         |
                                    |timestep +--------^                            |        +------------> Model   |
+--------+                          | (t)     |        |     +--------+    X'(n)    |        | |          |         |
|        |                          |         |        |     |        +----------------------+ |          +--^------+
| Model  +-------------------------->         |        +-----> Model  |             |          |             |
|        |                          |         |         X(t) |        +--+          |          |             |
+--------+                          |         |              +----^---+  |          |          |             |
+---------+                         +---------+    R(t+1)         |      |          +----------+             |
|  State  |                         |  State  +-------------------^    +-v----+     |  State   |             |
| M(0)    +------------------------->   M()   |    M(t+1)              | R(t) |     |  M(n)    +-------------+
| R(0)    |                         |   R()   |                        | M(t) |     |  R(n)    |
+---------+                         +-+--^----+                        +--+---+     +---^------+
                                      |  |                                |             |
                                      |  +--------------------------------+             |
                                      |                                                 |
                                      +-------------------------------------------------+

        Where:
        - M(t) is the memory tensor state at time t
        - R(t) is the residual tensor accumulator at time t
        - X'(n) is the predicted next step from the real X(n)
        - X(t) is the sequence data at time t
        - Y(t) is a sequence of subsequences created from X(t),
        with only some variables present (defined as 'key_variables')
        - `n` is the last element in the timeseries sequence
        - `t` is the timeseries step variable

        """

        tensor = convert_to_torch_tensor(data)
        init_residual = generate_state(self.residual_shape)
        init_memory = generate_state(self.memory_shape)
        last_step, checkpoint_residual, checkpoint_memory = self.update(init_residual, init_memory, tensor)
        raw_forecast = self.forecast_step(checkpoint_residual, checkpoint_memory, last_step)
        filtered_forecast = self.select_key_variables(raw_forecast)
        numpy_forecast = filtered_forecast.detach().numpy()
        return numpy_forecast

    def train_model(self, data: np.ndarray):

        r"""
        Our training regimine is different than most, as we forecast at _every_ timestep, rather than just at the end.

        The training process is described as below:

          +---------+------+
          |Data Set | X(t) |
          +----+----+------+
               |
               |
           +---v----+
           |        |
           v        v
          X(t)     Y(t)
           +        +
           |        |
           v        +
          X(t)     X(t:t+n)
           +        +
           |        |
           |        |
           +---+----+
               |
               |
               |
           +---v----+               +---------+                                                            Loss
           |        |               |         |         Y(t)                                                 ^
           |Training+--------------->         |        +-------------------------------------------+         |
           |Set     |               |         |        |                                           |         |
           +--------+               |For each |        |                                           |    +----+----+
                                    |timestep +-------->                X'(t)                      +---->         |
+--------+                          | (t)     |        |     +--------+    +    +----------+            |Criterion|
|        |                          |         |        |     |        +----+---->          |        +--->         |
| Model  +-------------------------->         |        +-----> Model  |         | Model    +--------+   +---------+
|        |                          |         |         X(t) |        +--+  +--->          |  H(t)
+--------+                          |         |              +----^---+  |  |   +----------+
+---------+                         +---------+    R(t+1)         |      |  |
|  State  |                         |  State  +-------------------^    +-v--+-+
| M(0)    +------------------------->   M()   |    M(t+1)              | R(t) |
| R(0)    |                         |   R()   |                        | M(t) |
+---------+                         +----^----+                        +--+---+
                                         |                                |
                                         +--------------------------------+

    Where:
     - M(t) is the memory tensor state at time t
     - R(t) is the residual tensor accumulator at time t
     - X'(t) is the predicted next step from the real X(t)
     - Y(t) is a sequence of subsequences created from X(t),
     with only some variables present (defined as 'key_variables')
     - H(t) is the attempted replication of Y(t), using X'(t) at time t
     - X(t) is the sequence data at time t
     - `t` is the timeseries step variable
        """

        tensor = convert_to_torch_tensor(data)
        criterion = nn.MSELoss()
        parameters = self.network.parameters()
        optimizer = optim.Adam(parameters, lr=35e-4)
        x, y = self.segment_data(tensor)
        start = perf_counter()
        total_time = 0
        while True:
            if total_time >= self.training_time:
                break
            optimizer.zero_grad()
            residual = generate_state(self.residual_shape)
            memory = generate_state(self.memory_shape)
            h, residual, memory = self.forecast_every_step(residual, memory, x)
            y_f = self.select_key_variables(y)
            h_f = self.select_key_variables(h)
            loss = criterion(h_f, y_f)
            loss_cpu = loss.item()
            print('training loss: {}'.format(str(loss_cpu)))
            loss.backward()
            optimizer.step()
            total_time = perf_counter() - start
            print("current training time: {}s".format(str(total_time)))
        print('best training loss: {}'.format(str(loss_cpu)))
        return loss_cpu

    def extract_network(self):
        return self.network

    def forecast_every_step(self, residual, memory, x):
        h = []
        for i in range(len(x)):
            x_t = x[i].view(1, -1)
            h_t, residual, memory = self.network(x_t, residual, memory)
            h_n = self.forecast_step(residual, memory, h_t[-1])
            h.append(h_n)
        h = torch.stack(h)
        return h, residual, memory

    def update(self, residual: torch.Tensor, memory: torch.tensor, x: torch.Tensor):
        h_t = None
        for i in range(len(x)):
            x_t = x[i].view(1, -1)
            x_t = self.noise.add_noise(x_t)
            h_t, residual, memory = self.network(x_t, residual, memory)
        return h_t, residual, memory

    def forecast_step(self, residual_t, memory_t, last_step):
        residual_forecast = residual_t.clone()
        memory_forecast = memory_t.clone()
        forecast_tensor = torch.zeros(self.forecast_length, self.data_dimensionality)
        forecast_tensor[0] = last_step
        for i in range(1, self.forecast_length):
            x_t = forecast_tensor[i - 1]
            x_t = self.noise.add_noise(x_t)
            next_step, residual_forecast, memory_forecast = self.network(x_t, residual_forecast, memory_forecast)
            forecast_tensor[i] = next_step
        return forecast_tensor

    def select_key_variables(self, tensor: torch.Tensor):
        if self.key_variables:
            filtered_tensors = []
            if len(tensor.shape) == 3:
                for feature in self.key_variables:
                    index = feature['index']
                    filtered_tensors.append(tensor[:, :, index])
                filtered_tensor = torch.stack(filtered_tensors, dim=2)
            else:
                for feature in self.key_variables:
                    index = feature['index']
                    filtered_tensors.append(tensor[:, index])
                filtered_tensor = torch.stack(filtered_tensors, dim=1)
        else:
            filtered_tensor = tensor
        return filtered_tensor

    def segment_data(self, data: torch.Tensor):
        segments = []
        for i in range(data.shape[0] - (self.forecast_length + 1)):
            segment = data[i + 1:i + self.forecast_length + 1]
            segments.append(segment)
        x = data[:-(self.forecast_length + 1)]
        y = torch.stack(segments)
        return x, y


def convert_to_torch_tensor(data: np.ndarray):
    return Variable(from_numpy(data)).float()


def generate_state(shape: tuple):
    tensor = torch.zeros(shape)
    return tensor


def init_network(architecture):
    network = ForecastNetwork(architecture).float()
    return network

from time import perf_counter

import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import from_numpy
from src.modules import forecast_model


class Model:
    def __init__(self, meta_data, network=None):
        self.residual_shape = meta_data['tensor_shape']['residual']
        self.memory_shape = meta_data['tensor_shape']['memory']
        self.data_dimensionality = meta_data['io_dimension']
        self.feature_columns = meta_data['feature_columns']
        self.forecast_length = meta_data['forecast_length']
        self.training_time = meta_data['training_time']
        if network:
            self.network = network
        else:
            self.network = init_network(meta_data['architecture'])



    def forecast(self, data: np.ndarray):
        tensor = convert_to_torch_tensor(data)
        init_residual = generate_state(self.residual_shape)
        init_memory = generate_state(self.memory_shape)
        last_step, checkpoint_residual, checkpoint_memory = self.update(init_residual, init_memory, tensor)
        raw_forecast = self.forecast_inner(checkpoint_residual, checkpoint_memory, last_step)
        filtered_forecast = self.filter_feature_cols(raw_forecast)
        numpy_forecast = filtered_forecast.detach().numpy()
        return numpy_forecast

    def train(self, data: np.ndarray):
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
            h, residual, memory = self.train_inner(residual, memory, x)
            y_f = self.filter_feature_cols(y)
            h_f = self.filter_feature_cols(h)
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

    def train_inner(self, residual, memory, x):
        h = []
        for i in range(len(x)):
            x_t = x[i].view(1, -1)
            h_t, residual, memory = self.network.forward(x_t, residual, memory)
            h_n = self.forecast_inner(residual, memory, h_t[-1])
            h.append(h_n)
        h = torch.stack(h)
        return h, residual, memory


    def update(self, residual: torch.Tensor, memory: torch.tensor, x: torch.Tensor):
        h_t = None
        for i in range(len(x)):
            x_t = x[i].view(1, -1)
            h_t, residual, memory = self.network.forward(x_t, residual, memory)
        return h_t, residual, memory



    def forecast_inner(self, residual_t, memory_t, h_t):
        residual_forecast = residual_t.clone()
        memory_forecast = memory_t.clone()
        forecast_tensor = torch.zeros(self.forecast_length, self.data_dimensionality)
        forecast_tensor[0] = h_t
        for i in range(1, self.forecast_length):
            output, residual_forecast, memory_forecast = self.network.forward(forecast_tensor[i - 1], residual_forecast,
                                                                              memory_forecast)
            forecast_tensor[i] = output
        return forecast_tensor


    def filter_feature_cols(self, tensor: torch.Tensor):
        if self.feature_columns:
            filtered_tensors = []
            if len(tensor.shape) == 3:
                for feature in self.feature_columns:
                    index = feature['index']
                    filtered_tensors.append(tensor[:, :, index])
                filtered_tensor = torch.stack(filtered_tensors, dim=2)
            else:
                for feature in self.feature_columns:
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
    return Variable(from_numpy(data), requires_grad=False).float()


def generate_state(shape: tuple):
    tensor = torch.zeros(shape)
    return tensor


def init_network(architecture):
    network = forecast_model.ForecastNetwork(architecture).float()
    return network

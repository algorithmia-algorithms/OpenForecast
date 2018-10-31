from time import perf_counter

import torch
from torch import nn
from torch import optim
from src.modules import forecast_model


class UtilitySchema():
    def __init__(self, meta_data):

        self.residual_shape = meta_data['tensor_shape']['residual']
        self.memory_shape = meta_data['tensor_shape']['memory']
        self.data_dimensionality = meta_data['io_dimension']
        self.feature_columns = meta_data['feature_columns']
        self.forecast_length = meta_data['forecast_length']
        self.training_time = meta_data['training_time']

#
def generate_forecast(model: forecast_model.ForecastModel, schema: UtilitySchema, data: torch.Tensor):
    init_residual = generate_state(schema.residual_shape)
    init_memory = generate_state(schema.memory_shape)
    last_step, checkpoint_residual, checkpoint_memory = update(model, init_residual, init_memory, data)
    raw_forecast = model_forecast(model, checkpoint_residual, checkpoint_memory, last_step, schema)
    filtered_forecast = filter_feature_cols(raw_forecast, schema)
    numpy_forecast = filtered_forecast.detach().numpy()
    return numpy_forecast


#
def train_model(model: forecast_model.ForecastModel, schema: UtilitySchema, data: torch.Tensor):
    criterion = nn.MSELoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=35e-4)
    x, y = segment_data(data, schema)
    start = perf_counter()
    total_time = 0
    while True:
        if total_time >= schema.training_time:
            break
        optimizer.zero_grad()
        residual = generate_state(schema.residual_shape)
        memory = generate_state(schema.memory_shape)
        h, residual, memory = train_all(model, residual, memory, x, schema)
        y_f = filter_feature_cols(y, schema)
        h_f = filter_feature_cols(h, schema)
        loss = criterion(h_f, y_f)
        loss_cpu = loss.item()
        print('training loss: {}'.format(str(loss_cpu)))
        loss.backward()
        optimizer.step()
        total_time = perf_counter() - start
        print("current training time: {}s".format(str(total_time)))
    print('best training loss: {}'.format(str(loss_cpu)))
    return model, loss_cpu


def train_all(model, residual: torch.Tensor, memory: torch.tensor, x: torch.Tensor, schema: UtilitySchema):
    h = []
    for i in range(len(x)):
        x_t = x[i].view(1, -1)
        h_t, residual, memory = training_step(model, x_t, residual, memory, schema)
        h.append(h_t)
    h = torch.stack(h)
    return h, residual, memory


def update(model: forecast_model.ForecastModel, residual: torch.Tensor, memory: torch.tensor, x: torch.Tensor):
    h_t = None
    for i in range(len(x)):
        x_t = x[i].view(1, -1)
        h_t, residual, memory = update_step(model, x_t, residual, memory)
    return h_t, residual, memory



# At every step, we forecast the next n points.
def training_step(model, x_t: torch.Tensor, residual_t: torch.Tensor, memory_t: torch.Tensor, schema: UtilitySchema):

    output_t, residual_t, memory_t = model.forward(x_t, residual_t, memory_t)
    forecast_t = model_forecast(model, residual_t, memory_t, output_t, schema)
    return forecast_t, residual_t, memory_t

# But when were
def update_step(model, x_t: torch.Tensor, residual_t: torch.Tensor, memory_t: torch.Tensor):
    output_t, residual_t, memory_t = model.forward(x_t, residual_t, memory_t)
    return output_t, residual_t, memory_t


def model_forecast(model: forecast_model.ForecastModel, residual_t, memory_t, h_t, schema: UtilitySchema):
    residual_forecast = clone_state(residual_t)
    memory_forecast = clone_state(memory_t)
    forecast_tensor = torch.zeros(schema.forecast_length, schema.data_dimensionality)
    forecast_tensor[0] = h_t
    for i in range(1, schema.forecast_length):
        output, residual_forecast, memory_forecast = model.forward(forecast_tensor[i - 1], residual_forecast,
                                                                   memory_forecast)
        forecast_tensor[i] = output
    return forecast_tensor


def filter_feature_cols(tensor: torch.Tensor, schema: UtilitySchema):
    if schema.feature_columns:
        filtered_tensors = []
        if len(tensor.shape) == 3:
            for feature in schema.feature_columns:
                    filtered_tensors.append(tensor[:, :, feature])
            filtered_tensor = torch.stack(filtered_tensors, dim=2)
        else:
            for feature in schema.feature_columns:
                    filtered_tensors.append(tensor[:, feature])
            filtered_tensor = torch.stack(filtered_tensors, dim=1)
    else:
        filtered_tensor = tensor
    return filtered_tensor


def clone_state(tensor):
    cloned = tensor.clone()
    return cloned


def generate_state(shape: tuple):
    tensor = torch.zeros(shape)
    return tensor


def init_network(architecture):
    network = forecast_model.ForecastModel(architecture).float()
    return network


def segment_data(data: torch.Tensor, schema: UtilitySchema):
    segments = []
    for i in range(data.shape[0]-(schema.forecast_length + 1)):
        segment = data[i+1:i+schema.forecast_length + 1]
        segments.append(segment)
    x = data[:-(schema.forecast_length + 1)]
    y = torch.stack(segments)
    return x, y
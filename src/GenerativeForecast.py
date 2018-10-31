#!/usr/bin/env python3
from src.modules import data_utilities, network
from src.modules import torch_utilities


class InputGuard:
    def __init__(self):
        self.mode = None
        self.data_path = None
        self.checkpoint_input_path = None
        self.checkpoint_output_path = None
        self.graph_save_path = None
        self.training_time = 500
        self.forecast_length = 5
        self.model_complexity = 0.5
        self.io_noise = 0.05



def process_input(input):
    guard = InputGuard()

    if 'mode' in input:
        if input['mode'] in ['forecast', 'train']:
            guard.mode = input['mode']
        else:
            raise network.AlgorithmError("mode is invalid, must be 'forecast', or 'train'")
        if 'data_path' in input:
            guard.data_path = type_check(input, 'data_path', str)
        else:
            raise network.AlgorithmError("'data_path' required")

        if 'forecast_length' in input:
            guard.forecast_length = type_check(input, 'forecast_length', int)
    if guard.mode == "train":
        if 'checkpoint_output_path' in input:
            guard.checkpoint_output_path = type_check(input, 'checkpoint_output_path', str)
        else:
            raise network.AlgorithmError("'checkpoint_output_path' required for training")
        if 'model_complexity' in input:
            guard.model_complexity = type_check(input, 'model_complexity', int)
        if 'training_time' in input:
                guard.training_time = type_check(input, 'training_time', int)
    elif guard.mode == "forecast":
        if 'checkpoint_input_path' in input:
            guard.checkpoint_input_path = type_check(input, 'checkpoint_input_path', str)
        else:
            raise network.AlgorithmError("'checkpoint_input_path' required for 'forecast' mode.")
        if 'graph_save_path' in input:
            guard.graph_save_path = type_check(input, 'graph_save_path', str)

    return guard



def type_check(dic, id, type):
    if isinstance(dic[id], type):
        return dic[id]
    else:
        raise network.AlgorithmError("'{}' must be of {}".format(str(id), str(type)))


def forecast(guard, outlier_removal_multiplier):
    output = dict()
    model, meta_data = network.get_model_package(guard.checkpoint_input_path)
    data_path = network.get_data(guard.data_path)
    data = network.load_json(data_path)
    data, meta_data = data_utilities.process_input(data, outlier_removal_multiplier, guard.model_complexity, guard.io_noise,
                                                   guard.forecast_length, guard.training_time, meta_data)
    schema = torch_utilities.UtilitySchema(meta_data)
    forecast_result = torch_utilities.generate_forecast(model, schema, data)
    output['forecast'] = data_utilities.format_forecast(forecast_result, meta_data)
    if guard.graph_save_path:
        local_graph_path = data_utilities.generate_graph(data, forecast_result, meta_data)
        output['graph_save_path'] = network.put_file(local_graph_path, guard.graph_save_path)

    return output


def train(guard, outlier_removal_multiplier):
    output = dict()
    meta_data = dict()
    data_path = network.get_data(guard.data_path)
    local_data = network.load_json(data_path)
    data, meta_data = data_utilities.process_input(local_data, outlier_removal_multiplier,
                                                   guard.model_complexity, guard.io_noise,
                                                   guard.forecast_length, guard.training_time,
                                                   meta_data)
    model = torch_utilities.init_network(meta_data['architecture'])
    schema = torch_utilities.UtilitySchema(meta_data)
    model, error = torch_utilities.train_model(model, schema, data)

    forecast_result = torch_utilities.generate_forecast(model, schema, data)
    output['forecast'] = data_utilities.format_forecast(forecast_result, meta_data['headers'])
    output['checkpoint_output_path'] = network.save_model_package(model, meta_data, guard.checkpoint_output_path)
    output['final_error'] = float(error)
    return output

def apply(input):
    guard = process_input(input)
    outlier_removal_multiplier = 15
    if guard.mode == "forecast":
        output = forecast(guard, outlier_removal_multiplier)
    else:
        output = train(guard, outlier_removal_multiplier)

    return output




def test_train():
    input = dict()
    input['mode'] = "train"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/rossman_5_training.json"
    # input['checkpoint_input_path'] = "data://timeseries/generativeforecasting/sinewave_v1.5_t0.t7"
    input['checkpoint_output_path'] = "data://timeseries/generativeforecasting/rossman_5.zip"
    input['training_time'] = 500
    input['complexity'] = 0.65
    input['forecast_length'] = 10
    return apply(input)


def test_forecast():
    input = dict()
    input['mode'] = "forecast"

    input['checkpoint_input_path'] = "data://timeseries/generativeforecasting/rossman_5.zip"
    input['graph_save_path'] = "data://timeseries/generativeforecasting/my_api_chart.png"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/rossman_5_training.json"
    input['forecast_length'] = 30
    input['iterations'] = 5
    input['io_noise'] = 0.05
    print(input)
    return apply(input)

if __name__ == "__main__":
  # result = test_forecast()
  result = test_train()
  print(result)


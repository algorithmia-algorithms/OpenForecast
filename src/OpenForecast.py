#!/usr/bin/env python3
from src.modules import data_utilities, utilities
from src.modules import model_manager


class Parameters:
    def __init__(self):
        self.mode = None
        self.data_path = None
        self.model_input_path = None
        self.model_output_path = None
        self.graph_save_path = None
        self.training_time = 500
        self.forecast_length = 10
        self.model_complexity = 0.5
        self.io_noise = 0.05
        self.outlier_removal_multiplier = 15



def process_input(input):
    parameters = Parameters()

    if 'outlier_removal_multiplier' in input:
        parameters.outlier_removal_multiplier = type_check(input, 'outlier_removal_multiplier', float)

    if 'data_path' in input:
        parameters.data_path = type_check(input, 'data_path', str)
    else:
        raise utilities.AlgorithmError("'data_path' required")

    if 'mode' in input:
        if input['mode'] in ['forecast', 'train']:
            parameters.mode = input['mode']
        else:
            raise utilities.AlgorithmError("mode is invalid, must be 'forecast', or 'train'")
    if parameters.mode == "train":
        if 'model_complexity' in input:
            parameters.model_complexity = type_check(input, 'model_complexity', float)
        if 'training_time' in input:
                parameters.training_time = type_check(input, 'training_time', int)
        if 'forecast_length' in input:
            parameters.forecast_length = type_check(input, 'forecast_length', int)
        if 'model_output_path' in input:
            parameters.model_output_path = type_check(input, 'model_output_path', str)
        else:
            raise utilities.AlgorithmError("'model_output_path' required for training")
    elif parameters.mode == "forecast":
        if 'graph_save_path' in input:
            parameters.graph_save_path = type_check(input, 'graph_save_path', str)
        if 'model_input_path' in input:
            parameters.model_input_path = type_check(input, 'model_input_path', str)
        else:
            raise utilities.AlgorithmError("'model_input_path' required for 'forecast' mode.")

    return parameters



def type_check(dic, id, type):
    if isinstance(dic[id], type):
        return dic[id]
    else:
        raise utilities.AlgorithmError("'{}' must be of {}".format(str(id), str(type)))




def forecast(input: Parameters):
    output = dict()
    network, meta_data = utilities.get_model_package(input.model_input_path)
    data_path = utilities.get_data(input.data_path)
    data = utilities.load_json(data_path)
    data, meta_data = data_utilities.process_input(data, input, meta_data)
    model = model_manager.Model(meta_data, network)
    forecast_result = model.forecast(data)
    output['forecast'] = data_utilities.format_forecast(forecast_result, meta_data)
    if input.graph_save_path:
        local_graph_path = data_utilities.generate_graph(data, forecast_result, meta_data)
        output['graph_save_path'] = utilities.put_file(local_graph_path, input.graph_save_path)

    return output


def train(input):
    output = dict()
    data_path = utilities.get_data(input.data_path)
    local_data = utilities.load_json(data_path)
    data, meta_data = data_utilities.process_input(local_data, input)
    model = model_manager.Model(meta_data)
    error = model.train_model(data)
    forecast_result = model.forecast(data)
    network = model.extract_network()
    output['forecast'] = data_utilities.format_forecast(forecast_result, meta_data)
    output['model_output_path'] = utilities.save_model_package(network, meta_data, input.model_output_path)
    output['final_error'] = float(error)
    return output

def apply(input):
    guard = process_input(input)
    if guard.mode == "forecast":
        output = forecast(guard)
    else:
        output = train(guard)

    return output




def test_train():
    input = dict()
    input['mode'] = "train"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/formatted_data_rossman_10.json"
    # input['model_input_path'] = "data://timeseries/generativeforecasting/sinewave_v1.5_t0.t7"
    input['model_output_path'] = "data://timeseries/generativeforecasting/rossman_10.zip"
    input['training_time'] = 300
    input['model_complexity'] = 0.65
    input['forecast_length'] = 10
    return apply(input)


def test_forecast():
    input = dict()
    input['mode'] = "forecast"

    input['model_input_path'] = "data://timeseries/generativeforecasting/rossman_10.zip"
    input['graph_save_path'] = "data://timeseries/generativeforecasting/my_api_chart.png"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/formatted_data_rossman_10.json"
    input['forecast_length'] = 15
    input['io_noise'] = 0.05
    print(input)
    return apply(input)

if __name__ == "__main__":
  # result = test_forecast()
  result = test_train()
  print(result)

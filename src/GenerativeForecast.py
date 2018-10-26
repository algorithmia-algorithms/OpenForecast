#!/usr/bin/env python3
from src.modules import data_utilities, graph, envelope, network
from src.modules import torch_utilities


class InputGuard:
    def __init__(self):
        self.mode = None
        self.data_path = None
        self.checkpoint_input_path = None
        self.checkpoint_output_path = None
        self.graph_save_path = None
        self.training_time = 500
        self.forecast_size = 15
        self.model_complexity = 0.5



def process_input(input):
    guard = InputGuard()

    if 'mode' in input:
        if input['mode'] in ['forecast', 'train']:
            guard.mode = input['mode']
        else:
            raise network.AlgorithmError("mode is invalid, must be 'forecast', or 'train'")
    if guard.mode == "train":
        if 'data_path' in input:
            guard.data_path = type_check(input, 'data_path', str)
        else:
            raise network.AlgorithmError("'data_path' required for 'train' mode")
        if 'checkpoint_input_path' in input:
            guard.checkpoint_input_path = type_check(input, 'checkpoint_input_path', str)
        if 'checkpoint_output_path' in input:
            guard.checkpoint_output_path = type_check(input, 'checkpoint_output_path', str)
        else:
            raise network.AlgorithmError("'checkpoint_output_path' required for training")
        if 'model_complexity' in input:
            guard.model_complexity = type_check(input, 'model_complexity', int)
        if 'training_time' in input:
                guard.training_time = type_check(input, 'training_time', int)
    elif guard.mode == "forecast":
        if 'data_path' in input:
            guard.data_path = type_check(input, 'data_path', str)
        if 'checkpoint_input_path' in input:
            guard.checkpoint_input_path = type_check(input, 'checkpoint_input_path', str)
        else:
            raise network.AlgorithmError("'checkpoint_input_path' required for 'forecast' mode.")
        if 'checkpoint_output_path' in input:
            guard.checkpoint_output_path = type_check(input, 'checkpoint_output_path', str)
        if 'forecast_size' in input:
            guard.forecast_size = int(input['forecast_size'])
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
    model, meta_data, state = network.get_model_package(guard.checkpoint_input_path)
    if guard.data_path:
        data = network.get_data(guard.data_path)
        data, norm_boundaries, headers, feature_columns = data_utilities.process_input(data, outlier_removal_multiplier)
    forecast_result, state = torch_utilities.create_forecasts(guard.data_path, model, state,
                                                                            guard.training_time, guard.forecast_size)

    # output_env = envelope.create_envelope(raw_forecast, guard.forecast_size)
    # if guard.graph_save_path:
    #     graphing_env = envelope.create_envelope(normal_forecast, guard.forecast_size)
    #     graph_path = graph.create_graph(graphing_env, state, guard.forecast_size, guard.io_noise)
    #     output['saved_graph_path'] = graph.save_graph(graph_path, guard.graph_save_path)
    # if guard.checkpoint_output_path:
    #     output['checkpoint_output_path'] = data_utilities.save_network_to_algo(model, guard.checkpoint_output_path)
    # formatted_envelope = envelope.ready_envelope(output_env, state)
    # output['envelope'] = formatted_envelope
    return output


def train(guard, outlier_removal_multiplier):
    output = dict()
    local_data_path = network.get_data(guard.data_path)
    local_data = network.load_json(local_data_path)
    if guard.checkpoint_input_path:
        model, meta_data, state = network.get_model_package(guard.checkpoint_input_path)
        data, norm_boundaries, headers, feature_columns = data_utilities.process_input(local_data,
                                                                                       outlier_removal_multiplier)
    else:
        data, norm_boundaries, headers, feature_columns = data_utilities.process_input(local_data,
                                                                                  outlier_removal_multiplier)
        io_dim = len(norm_boundaries)
        model, meta_data, state = torch_utilities.init_network(io_width=io_dim, io_noise=0.04, complexity=guard.model_complexity,
                                                      headers=headers, feature_columns=feature_columns)
    meta_data['training_time'] = guard.training_time
    error, checkpoint = torch_utilities.train_autogenerative_model(data_frame=data, network=model,
                                                                   state=state, meta_data=meta_data)
    output['checkpoint_output_path'], output['state_output_path'] = data.save_network_to_algo(checkpoint, state, guard.checkpoint_output_path)
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
    return apply(input)


def test_forecast():
    input = dict()
    input['mode'] = "forecast"

    input['checkpoint_input_path'] = "data://timeseries/generativeforecasting/apidata_v1.0.1_t0.zip"
    input['graph_save_path'] = "data://timeseries/generativeforecasting/my_api_chart.png"
    # input['data_path'] = "data://TimeSeries/GenerativeForecasting/apidata_v0.2.5_t1.csv"
    input['forecast_size'] = 500
    input['iterations'] = 5
    input['io_noise'] = 0.05
    print(input)
    return apply(input)

if __name__ == "__main__":
  # result = test_forecast()
  result = test_train()
  print(result)


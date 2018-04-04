#!/usr/bin/env python3
from src.modules import data_proc, graph, envelope, misc
from src.modules import network_processing
cuda = False


class InputGuard:
    def __init__(self):
        self.mode = None
        self.data_path = None
        self.checkpoint_input_path = None
        self.checkpoint_output_path = None
        self.graph_save_path = None
        self.iterations = 10
        self.forecast_size = 15
        self.layer_width = 51
        self.io_noise = 0.04
        self.attention_beam_width = 45
        self.future_beam_width = 1
        self.input_dropout = 0.45



def process_input(input):
    guard = InputGuard()

    if 'mode' in input:
        if input['mode'] in ['forecast', 'train']:
            guard.mode = input['mode']
        else:
            raise misc.AlgorithmError("mode is invalid, must be 'forecast', or 'train'")
    if guard.mode == "train":
        if 'data_path' in input:
            guard.data_path = type_check(input, 'data_path', str)
        else:
            raise misc.AlgorithmError("'data_path' required for 'train' mode")
        if 'checkpoint_input_path' in input:
            guard.checkpoint_input_path = type_check(input, 'checkpoint_input_path', str)
        if 'checkpoint_output_path' in input:
            guard.checkpoint_output_path = type_check(input, 'checkpoint_output_path', str)
        else:
            raise misc.AlgorithmError("'checkpoint_output_path' required for 'train' mode")
        if 'hidden_width' in input:
            guard.layer_width = type_check(input, 'hidden_width', int)
        if 'attention_beam_width' in input:
            guard.attention_beam_width = type_check(input, 'attention_beam_width', int)
        if 'future_beam_width' in input:
            guard.future_beam_width = type_check(input, 'future_beam_width', int)
        if 'input_dropout' in input:
            guard.input_dropout = type_check(input, 'input_dropout', float)
        if 'iterations' in input:
                guard.iterations = type_check(input, 'iterations', int)
        if 'io_noise' in input:
                guard.io_noise = type_check(input, 'io_noise', float)
    elif guard.mode == "forecast":
        if 'data_path' in input:
            guard.data_path = type_check(input, 'data_path', str)
        if 'checkpoint_input_path' in input:
            guard.checkpoint_input_path = type_check(input, 'checkpoint_input_path', str)
        else:
            raise misc.AlgorithmError("'checkpoint_input_path' required for 'forecast' mode.")
        if 'checkpoint_output_path' in input:
            guard.checkpoint_output_path = type_check(input, 'checkpoint_output_path', str)
        if 'forecast_size' in input:
            guard.forecast_size = int(input['forecast_size'])
        if 'graph_save_path' in input:
            guard.graph_save_path = type_check(input, 'graph_save_path', str)
        if 'iterations' in input:
                guard.iterations = type_check(input, 'iterations', int)
        if 'io_noise' in input:
                guard.io_noise = type_check(input, 'io_noise', float)

    return guard



def type_check(dic, id, type):
    if isinstance(dic[id], type):
        return dic[id]
    else:
        raise misc.AlgorithmError("'{}' must be of {}".format(str(id), str(type)))


def forecast(guard, outlier_removal_multiplier):
    output = dict()
    network, state = data_proc.load_network_from_algo(guard.checkpoint_input_path)
    if guard.data_path:
        guard.data_path = data_proc.get_frame(guard.data_path)
        guard.data_path = data_proc.process_sequence_incremental(guard.data_path, state,
                                                                 outlier_removal_multiplier)

    normal_forecast, raw_forecast, state = network_processing.create_forecasts(guard.data_path, network, state,
                                                                               guard.iterations, guard.forecast_size,
                                                                               guard.io_noise)

    output_env = envelope.create_envelope(raw_forecast, guard.forecast_size)
    if guard.graph_save_path:
        graphing_env = envelope.create_envelope(normal_forecast, guard.forecast_size)
        graph_path = graph.create_graph(graphing_env, state, guard.forecast_size, guard.io_noise)
        output['saved_graph_path'] = graph.save_graph(graph_path, guard.graph_save_path)
    if guard.checkpoint_output_path:
        output['checkpoint_output_path'] = data_proc.save_network_to_algo(network, guard.checkpoint_output_path)
    formatted_envelope = envelope.ready_envelope(output_env, state)
    output['envelope'] = formatted_envelope
    return output


def train(guard, max_history, base_learning_rate, outlier_removal_multiplier, gradient_multiplier):
    output = dict()
    guard.data_path = data_proc.get_frame(guard.data_path)
    if guard.checkpoint_input_path:
        network, state = data_proc.load_network_from_algo(guard.checkpoint_input_path)
        data = data_proc.process_sequence_incremental(guard.data_path, state, outlier_removal_multiplier)
        # lr_rate = network_processing.determine_lr(data, state)
    else:
        data, norm_boundaries, headers = data_proc.process_sequence_initial(guard.data_path,
                                                                            outlier_removal_multiplier)
        io_dim = len(norm_boundaries)
        learning_rate = float(base_learning_rate) / io_dim
        network, state = network_processing.initialize_network(io_dim=io_dim, layer_width=guard.layer_width,
                                                               max_history=max_history,
                                                               initial_lr=learning_rate, lr_multiplier=gradient_multiplier,
                                                               io_noise=guard.io_noise,
                                                               training_length=data['x'].shape[0],
                                                               attention_beam_width=guard.attention_beam_width, headers=headers)
        network.initialize_meta(len(data['x']), norm_boundaries)
        # lr_rate = state['prime_lr']
    error, network = network_processing.train_autogenerative_model(data_frame=data, network=network,
                                                                   checkpoint_state=state, iterations=guard.iterations)
    output['checkpoint_output_path'] = data_proc.save_network_to_algo(network, guard.checkpoint_output_path)
    output['final_error'] = float(error)
    return output

def apply(input):
    guard = process_input(input)
    outlier_removal_multiplier = 15
    max_history = 500
    base_learning_rate = 0.5
    gradient_multiplier = 1.0
    if guard.mode == "forecast":
        output = forecast(guard, outlier_removal_multiplier)
    else:
        output = train(guard, max_history, base_learning_rate, outlier_removal_multiplier, gradient_multiplier)

    return output




def test_train():
    input = dict()
    input['mode'] = "train"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/apidata_v0.2.5_t0.csv"
    # input['checkpoint_input_path'] = "data://timeseries/generativeforecasting/sinewave_v1.5_t0.t7"
    input['checkpoint_output_path'] = "data://timeseries/generativeforecasting/apidata_v0.0_t0.zip"
    input['iterations'] = 20
    input['io_noise'] = 0.04
    input['hidden_width'] = 100
    input['future_beam_width'] = 1
    input['attention_beam_width'] = 50
    return apply(input)


def test_forecast():
    input = dict()
    input['mode'] = "forecast"

    input['checkpoint_input_path'] = "data://timeseries/generativeforecasting/apidata_v0.0_t0.zip"
    input['graph_save_path'] = "data://timeseries/generativeforecasting/my_sinegraph.png"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/apidata_v0.2.5_t1.csv"
    input['forecast_size'] = 100
    input['iterations'] = 20
    input['io_noise'] = 0.04
    print(input)
    return apply(input)

if __name__ == "__main__":
  result = test_forecast()
  # result = test_train()
  print(result)

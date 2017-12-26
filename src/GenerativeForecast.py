import torch
from src.modules import data_proc, graph, misc, net_misc, envelope


class InputGuard():
    def __init__(self):
        self.mode = None
        self.data_path = None
        self.checkpoint_input_path = None
        self.checkpoint_output_path = None
        self.graph_save_path = None
        self.iterations = 10
        self.epochs = 4
        self.forecast_size = 15
        self.layer_width = 51
        self.io_noise = 0.04
        self.attention_beam_width = 25
        self.future_beam_width = 10
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
        if 'layer_width' in input:
            guard.layer_width = type_check(input, 'layer_width', int)
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

def apply(input):
    guard = process_input(input)
    outlier_removal_multiplier = 5
    max_history = 500
    base_learning_rate = 0.5
    gradient_multiplier = 1.0
    output = dict()
    if input['mode'] == "forecast":
        local_file = data_proc.get_file(guard.checkpoint_input_path)
        network, state = net_misc.load_checkpoint(local_file)
        if guard.data_path:
            guard.data_path = data_proc.get_frame(guard.data_path)
            guard.data_path = data_proc.process_frames_incremental(guard.data_path, state, outlier_removal_multiplier)

        normal_forecast, raw_forecast, state, checkpoint_model = net_misc.create_forecasts(guard.data_path, network, state, guard.iterations, guard.forecast_size, guard.io_noise)

        output_env = envelope.create_envelope(normal_forecast, guard.forecast_size, state)
        if guard.graph_save_path:
            graphing_env = envelope.create_envelope(raw_forecast, guard.forecast_size, state)
            graph_path = graph.create_graph(graphing_env, state, guard.forecast_size, guard.io_noise)
            output['saved_graph_path'] = graph.save_graph(graph_path, guard.graph_save_path)
        if guard.checkpoint_output_path:
            output['checkpoint_output_path'] = net_misc.save_model(checkpoint_model, guard.checkpoint_output_path)
        formatted_envelope = envelope.ready_envelope(output_env, state)
        output['envelope'] = formatted_envelope
    if guard.mode == "train":
        guard.data_path = data_proc.get_frame(guard.data_path)
        if guard.checkpoint_input_path:
            local_file = data_proc.get_file(guard.checkpoint_input_path)
            network, state = net_misc.load_checkpoint(local_file)
            data = data_proc.process_frames_incremental(guard.data_path, state, gradient_multiplier)
            lr_rate = net_misc.determine_lr(data, state)
        else:
            data, norm_boundaries, headers = data_proc.process_frames_initial(guard.data_path, outlier_removal_multiplier, beam_width=guard.future_beam_width)
            io_dim = len(norm_boundaries)
            learning_rate = float(base_learning_rate) / io_dim
            network, state = net_misc.initialize_network(io_dim=io_dim, layer_width=guard.layer_width, max_history=max_history,
                                                         initial_lr=learning_rate, lr_multiplier=gradient_multiplier,
                                                         io_noise=guard.io_noise, attention_beam_width=guard.attention_beam_width,
                                                         future_beam_width=guard.future_beam_width, headers=headers)
            network.initialize_meta(len(data['x']), norm_boundaries)
            lr_rate = state['prime_lr']
        error, network = net_misc.train_autogenerative_model(data_frame=data, network=network,
                                                             checkpoint_state=state, iterations=guard.iterations,
                                                             learning_rate=lr_rate, epochs=guard.epochs, drop_percentage=guard.input_dropout)
        output['checkpoint_output_path'] = net_misc.save_model(network, guard.checkpoint_output_path)
        output['final_error'] = float(error)
    return output


def test_train():
    input = dict()
    input['mode'] = "train"
    # input['mode'] = "forecast"
    # input['data_path'] = "data://zeryx/forecasting_testing/rossman_training_2.csv"
    # input['data_path'] = 'data://zeryx/forecasting_testing/sinewave_bulk.csv'
    # input['data_path'] = 'data://zeryx/forecasting_testing/sinewave_incremental.csv'
    # input['data_path'] = "data://TimeSeries/GenerativeForecasting/btc-train_2.csv"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/sinewave_bulk.csv"
    # input['data_path'] = 'data://zeryx/forecasting_testing/csdisco-train_1.csv'
    # input['data_path'] = 'data://zeryx/forecasting_testing/sine_wave_train.csv'
    # input['data_path'] = "data://zeryx/forecasting_testing/funpokes-train.csv"
    # input['data_path'] = "data://zeryx/forecasting_testing/funpokes-test.csv"
    # input['data_path'] = "data://zeryx/forecasting_testing/csdisco_2.csv"
    # input['data_path'] = "data://zeryx/forecasting_testing/csdisco_3.csv"
    # input['checkpoint_output_path'] = "data://zeryx/forecasting_testing/rossman_model.t7"
    # input['checkpoint_output_path'] = "data://zeryx/forecasting_testing/sinewave_bulk_model.t7"
    # input['checkpoint_output_path'] = "data://zeryx/forecasting_testing/sinewave_incremented_model.t7"
    # input['checkpoint_input_path'] = "data://zeryx/forecasting_testing/sinewave_bulk_model.t7"
    input['checkpoint_output_path'] = "data://timeseries/generativeforecasting/sinewave_headers.t7"
    # input['checkpoint_output_path'] = "data://zeryx/forecasting_testing/sine_model.t7"
    # input['checkpoint'] = "data://zeryx/forecasting_testing/trained_batch_model.t7"
    # input['checkpoint_output_path'] = "data://zeryx/forecasting_testing/unavariate_model.t7"
    # input['checkpoint_output_path'] = "data://zeryx/forecasting_testing/bivariate_model.t7"
    input['iterations'] = 2
    input['layer_width'] = 55
    input['io_noise'] = 0.04
    input['attention_beam_width'] = 55
    input['future_beam_width'] = 25
    input['input_dropout'] = 0.4
    print(input)
    return apply(input)


def test_forecast():
    input = dict()
    input['mode'] = "forecast"
    # input['checkpoint_input_path'] = "data://timeseries/generativeforecasting/btc_model_headers.t7"
    # input['data_path'] = "data://zeryx/forecasting_testing/funpokes-test.csv"
    # input['data_path'] = 'data://zeryx/forecasting_testing/sinewave_eval.csv'
    # input['data_path'] = 'data://zeryx/forecasting_testing/csdisco-test_2.csv'
    input['checkpoint_input_path'] = "data://timeseries/generativeforecasting/sinewave_headers.t7"
    # input['checkpoint_input_path'] = "data://zeryx/forecasting_testing/csdisco_model.t7"
    # input['data_path'] = 'data://zeryx/forecasting_testing/btc-test_2.csv'
    # input['data_path'] = 'data://zeryx/forecasting_testing/sine_wave_test.csv'
    # input['checkpoint_input_path'] = "data://zeryx/forecasting_testing/csdisco_model.t7"
    # input['checkpoint_output_path'] = "data://zeryx/forecasting_testing/unavariate_model_up2date.t7"
    # input['checkpoint_input_path'] = "data://zeryx/forecasting_testing/btc_up2date_model.t7"
    # input['checkpoint_input_path'] = "data://zeryx/forecasting_testing/sine_model.t7"
    # input['checkpoint_input_path'] = "data://zeryx/forecasting_testing/btc_model.t7"
    # input['checkpoint_output_path'] = "data://zeryx/forecasting_testing/bivariate_model_up2date.t7"
    # input['checkpoint_input_path'] = "data://zeryx/forecasting_testing/sinewave_incremented_model.t7"
    # input['checkpoint_input_path'] = "data://timeseries/generativeforecasting/btc_model_headers.t7"
    input['graph_save_path'] = "data://timeseries/generativeforecasting/my_sinegraph.png"
    input['forecast_size'] = 10
    input['iterations'] = 25
    input['io_noise'] = 0.05
    return apply(input)

torch.backends.cudnn.enabled = False

if __name__ == "__main__":
   result = test_forecast()
   # result = test_train()
   print(result)

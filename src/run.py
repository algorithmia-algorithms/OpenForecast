if __name__ == "__main__":
    import torch
    from src.modules import data_proc, graph, net_misc, envelope, misc
    import json
    import sys
    torch.backends.cudnn.enabled = False


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
            
        def __init__(self, dictionary):
            for key in dictionary:
                setattr(self, key, dictionary[key])

    def forecast(guard, outlier_removal_multiplier):
        output = dict()
        local_file = misc.get_file(guard.checkpoint_input_path)
        network, state = net_misc.load_checkpoint(local_file)
        if guard.data_path:
            guard.data_path = data_proc.get_frame(guard.data_path)
            guard.data_path = data_proc.process_sequence_incremental(guard.data_path, state,
                                                                     outlier_removal_multiplier)

        normal_forecast, raw_forecast, state = net_misc.create_forecasts(guard.data_path, network, state,
                                                                         guard.iterations, guard.forecast_size,
                                                                         guard.io_noise)

        output_env = envelope.create_envelope(normal_forecast, guard.forecast_size, state)
        if guard.graph_save_path:
            graphing_env = envelope.create_envelope(raw_forecast, guard.forecast_size, state)
            graph_path = graph.create_graph(graphing_env, state, guard.forecast_size, guard.io_noise)
            output['saved_graph_path'] = graph.save_graph(graph_path, guard.graph_save_path)
        if guard.checkpoint_output_path:
            output['checkpoint_output_path'] = net_misc.save_model(network, guard.checkpoint_output_path)
        formatted_envelope = envelope.ready_envelope(output_env, state)
        output['envelope'] = formatted_envelope
        return output

    def train(guard, max_history, base_learning_rate, outlier_removal_multiplier, gradient_multiplier):
        output = dict()
        guard.data_path = data_proc.get_frame(guard.data_path)
        if guard.checkpoint_input_path:
            local_file = misc.get_file(guard.checkpoint_input_path)
            network, state = net_misc.load_checkpoint(local_file)
            data = data_proc.process_sequence_incremental(guard.data_path, state, outlier_removal_multiplier)
            lr_rate = net_misc.determine_lr(data, state)
        else:
            data, norm_boundaries, headers = data_proc.process_sequence_initial(guard.data_path,
                                                                                outlier_removal_multiplier,
                                                                                beam_width=guard.future_beam_width)
            io_dim = len(norm_boundaries)
            learning_rate = float(base_learning_rate) / io_dim
            network, state = net_misc.initialize_network(io_dim=io_dim, layer_width=guard.layer_width,
                                                         max_history=max_history,
                                                         initial_lr=learning_rate, lr_multiplier=gradient_multiplier,
                                                         io_noise=guard.io_noise,
                                                         attention_beam_width=guard.attention_beam_width,
                                                         future_beam_width=guard.future_beam_width, headers=headers)
            network.initialize_meta(len(data['x']), norm_boundaries)
            lr_rate = state['prime_lr']
        error, network = net_misc.train_autogenerative_model(data_frame=data, network=network,
                                                             checkpoint_state=state, iterations=guard.iterations,
                                                             learning_rate=lr_rate, epochs=guard.epochs,
                                                             drop_percentage=guard.input_dropout)
        output['checkpoint_output_path'] = net_misc.save_model(network, guard.checkpoint_output_path)
        output['final_error'] = float(error)
        return output


    def run(guard):
        outlier_removal_multiplier = 5
        max_history = 500
        base_learning_rate = 0.5
        gradient_multiplier = 1.0
        if guard.mode == "forecast":
            output = forecast(guard, outlier_removal_multiplier)
        else:
            output = train(guard, max_history, base_learning_rate, outlier_removal_multiplier, gradient_multiplier)
        return output
    
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    with open(input_filename) as f:
        input = InputGuard(json.loads(f.read()))
    output = run(input)
    with open(output_filename, 'w') as f:
        json.dump(output, f)
    print('done')
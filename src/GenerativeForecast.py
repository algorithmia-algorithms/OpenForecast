import json
from src.modules import misc
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import tempfile, os
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

def apply(input):
    guard = process_input(input)
    output = execute_workaround(guard)
    return output


def execute_workaround(input_data):
    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libgfortran.so.3'
    _, in_filename = tempfile.mkstemp()
    _, out_filename = tempfile.mkstemp()
    print(in_filename)
    print(out_filename)
    with open(in_filename, 'w') as f:
        json.dump(input_data.__dict__, f)
    runShellCommand(['python', 'run.py', in_filename, out_filename], cwd=os.path.dirname(os.path.realpath(__file__)))
    with open(out_filename) as f:
        output = json.load(f)
    return output



def runShellCommand(commands, cwd=None):
    try:
        p = Popen(commands, stdout=PIPE, stderr=PIPE, cwd=cwd)
        output, error = p.communicate()
        if error:
            raise misc.AlgorithmError(error.strip())
    except CalledProcessError as e:
        raise e
    except misc.AlgorithmError as e:
        raise e


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
    input['checkpoint_output_path'] = "data://timeseries/generativeforecasting/sinewave_v1.0_t0.t7"
    # input['checkpoint_output_path'] = "data://timeseries/generativeforecasting/sinewave_v1.0_t1.t7"
    # input['checkpoint_output_path'] = "data://zeryx/forecasting_testing/sine_model.t7"
    # input['checkpoint'] = "data://zeryx/forecasting_testing/trained_batch_model.t7"
    # input['checkpoint_output_path'] = "data://zeryx/forecasting_testing/unavariate_model.t7"
    # input['checkpoint_output_path'] = "data://zeryx/forecasting_testing/bivariate_model.t7"
    input['iterations'] = 10
    # input['layer_width'] = 55
    input['io_noise'] = 0.06
    # input['attention_beam_width'] = 55
    # input['future_beam_width'] = 20
    input['input_dropout'] = 0.45
    return apply(input)


def test_forecast():
    input = dict()
    input['mode'] = "forecast"
    # input['checkpoint_input_path'] = "data://timeseries/generativeforecasting/btc_model_headers.t7"
    # input['data_path'] = "data://zeryx/forecasting_testing/funpokes-test.csv"
    # input['data_path'] = 'data://timeseries/generativeforecasting/sinewave_incremental.csv'
    # input['data_path'] = 'data://zeryx/forecasting_testing/csdisco-test_2.csv'
    # input['checkpoint_output_path'] = "data://timeseries/generativeforecasting/sinewave_v1.0_t0.t7"
    input['checkpoint_input_path'] = "data://timeseries/generativeforecasting/sinewave_v1.0_t1_inf.t7"
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
    input['forecast_size'] = 100
    input['iterations'] = 25
    input['io_noise'] = 0.06
    print(input)
    return apply(input)

# if __name__ == "__main__":
#   result = test_forecast()
#   # result = test_train()
#   print(result)

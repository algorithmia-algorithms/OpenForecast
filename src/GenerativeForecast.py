from src.modules import misc
import os, json, tempfile
from subprocess import Popen, PIPE
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

def execute_workaround(input_data):
    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libgfortran.so.3'
    os.environ['LC_ALL'] = 'C'
    _, in_filename = tempfile.mkstemp()
    _, out_filename = tempfile.mkstemp()
    print(in_filename)
    print(out_filename)
    with open(in_filename, 'w') as f:
        json.dump(input_data.__dict__, f)
    root_path = '/'.join(os.path.realpath(__file__).split('/')[:-2])
    print(root_path)
    # runShellCommand(['/opt/anaconda3/bin/python', 'src/run.py', in_filename, out_filename], cwd=root_path)
    runShellCommand(['python3', 'src/run.py', in_filename, out_filename], cwd=root_path)
    with open(out_filename) as f:
        output = json.load(f)
    return output

def runShellCommand(commands, cwd=None):
    p = Popen(commands, stdout=PIPE, stderr=PIPE, cwd=cwd)
    output, error = p.communicate()
    out_str = str(output.decode('utf-8'))
    err_str = str(error.decode('utf-8'))
    print(out_str)
    if error:
        raise misc.AlgorithmError(err_str)


def apply(input):
    guard = process_input(input)
    output = execute_workaround(guard)

    return output

def test_train():
    input = dict()
    input['mode'] = "train"
    # input['mode'] = "forecast"
    # input['data_path'] = "data://zeryx/forecasting_testing/rossman_training_2.csv"
    # input['data_path'] = 'data://zeryx/forecasting_testing/sinewave_bulk.csv'
    # input['data_path'] = 'data://zeryx/forecasting_testing/sinewave_incremental.csv'
    # input['data_path'] = "data://TimeSeries/GenerativeForecasting/btc-train_2.csv"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/sinewave_v1.0_t0.csv"
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
    input['iterations'] = 20
    input['io_noise'] = 0.06
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
    input['checkpoint_input_path'] = "data://timeseries/generativeforecasting/sinewave_v5.0_t0.t7"
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

#if __name__ == "__main__":
  # result = test_forecast()
  #result = test_train()
  #print(result)

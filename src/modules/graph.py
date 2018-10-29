from uuid import uuid4
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.modules import network


def test_graph(target, pred):
    seq_length = target.shape[0]
    x = np.arange(1, seq_length+1)
    plt.plot(x, pred, linestyle='--')
    plt.plot(x, target[:, :], linestyle='-')
    plt.show()
    print("shown")
    plt.close()

def create_graph(envelope, state, forecast_length, noise_percentage):
    graph_file_name = "/tmp/{}.png".format(str(uuid4()))
    pred_history = state['pred_history'].view(-1, state['io_width']).cpu().data.numpy()
    true_history = state['true_history'].view(-1, state['io_width']).cpu().data.numpy()
    headers = state['headers']
    if true_history.shape[0] <= forecast_length:
        true_past = true_history
        pred_past = pred_history
        gold_length = len(true_history)
    else:
        gold_length = forecast_length * 2
        true_past = true_history[-gold_length:]
        pred_past = pred_history[-gold_length:]

    gold_range = np.arange(gold_length)
    forecast_range = np.arange(gold_length, gold_length + forecast_length)
    first_up = envelope['first_deviation']['upper_bound']
    first_low = envelope['first_deviation']['lower_bound']
    second_up = envelope['second_deviation']['upper_bound']
    second_low = envelope['second_deviation']['lower_bound']
    for i in range(pred_history.shape[1]):
        history_color = np.random.rand(3,)
        plt.plot(gold_range, true_past[:, i], c=history_color, linestyle='-', label='historical real: {}'.format(headers[i]), linewidth=2.0)
        plt.plot(gold_range, pred_past[:, i], c=history_color, linestyle='--', label='historical predicted: {}'.format(headers[i]), linewidth=2.0)
        forecast_color = np.random.rand(3, )
        plt.plot(forecast_range, first_up[:, i], c=forecast_color, linestyle='--', label='1 sigma forecast: {}'.format(headers[i]), linewidth=1.5)
        plt.plot(forecast_range, first_low[:, i], c=forecast_color, linestyle='--', linewidth=1.5)
        plt.plot(forecast_range, second_up[:, i], c=forecast_color, linestyle=':', label='2 sigma forecast: {}'.format(headers[i]), linewidth=1.5)
        plt.plot(forecast_range, second_low[:,  i], c=forecast_color, linestyle=':', linewidth=1.5)

    plt.title("monte carlo forecast envelope, {}% halt_noise".format(noise_percentage*100))
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend(loc=2, fontsize='x-small')
    plt.savefig(graph_file_name)
    # plt.show()
    plt.close()
    return graph_file_name


def graph_training_data(forecast, forecast_target, historical_forecast, historical_input):
    historical_length = historical_forecast.shape[1]
    forecast_length = forecast.shape[1]
    forecast_target = forecast_target.cpu().data.numpy()
    historical_forecast = historical_forecast.cpu().data.numpy()
    historical_truth = historical_input.cpu().data.numpy()
    forecast = forecast.cpu().data.numpy()
    for i in range(forecast.shape[2]):
        hist_truth = historical_truth[:, i]
        forc_truth = forecast_target[:, i]
        plt.plot(np.arange(historical_length), hist_truth, c='y', linestyle='-')
        plt.plot(np.arange(historical_length, historical_length + forecast_length), forc_truth, c='y', linestyle='--', linewidth=2.0)
        for j in range(forecast.shape[0]):
            color = np.random.rand(3, )
            hist_forc = historical_forecast[j, :, i]
            forc = forecast[j, :, i]
            plt.plot(np.arange(j, historical_length+j), hist_forc, c=color, linestyle='-')
            plt.plot(np.arange(historical_length+j, historical_length+forecast_length+j), forc, c=color, linestyle=':', label='step {}'.format(str(j+1)))
    plt.legend()
    plt.show()
    plt.close()


def save_graph(graph_path, remote_url):
    return network.put_file(graph_path, remote_url)
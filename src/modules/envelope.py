import numpy as np


def create_envelope(forecasts, future_length, state):
    envelope = {'first_deviation': {'lower_bound': [], 'upper_bound': []}, 'second_deviation': {'lower_bound': [], 'upper_bound': []}, 'mean': [], 'standard_deviation': []}
    for i in range(future_length):
        step = []
        for j in range(forecasts.shape[1]):
            step.append(forecasts[i, j])
        step = np.asarray(step)
        mean = np.mean(step, axis=0)
        sd = np.std(step, axis=0)
        # mean = np.reshape(mean, (1, forecasts.shape[2]))
        # sd = sd, (1, forecasts.shape[2]))
        envelope['mean'].append(mean)
        envelope['standard_deviation'].append(sd)
        envelope['first_deviation']['upper_bound'].append(mean + sd)
        envelope['first_deviation']['lower_bound'].append(mean - sd)
        envelope['second_deviation']['upper_bound'].append(mean + 2 * sd)
        envelope['second_deviation']['lower_bound'].append(mean - 2 * sd)

    envelope['mean'] = np.asarray(envelope['mean'])
    envelope['standard_deviation'] = np.asarray(envelope['standard_deviation'])
    envelope['first_deviation']['upper_bound'] = np.asarray(envelope['first_deviation']['upper_bound'])
    envelope['first_deviation']['lower_bound'] = np.asarray(envelope['first_deviation']['lower_bound'])
    envelope['second_deviation']['upper_bound'] = np.asarray(envelope['second_deviation']['upper_bound'])
    envelope['second_deviation']['lower_bound'] = np.asarray(envelope['second_deviation']['lower_bound'])

    return envelope


def ready_envelope(raw_envelope, state):
    io_width = state['io_width']
    headers = state['headers']
    formatted_envelope = []
    mean = raw_envelope['mean']
    sd = raw_envelope['standard_deviation']
    first_upper = raw_envelope['first_deviation']['upper_bound']
    first_lower = raw_envelope['first_deviation']['lower_bound']
    second_upper = raw_envelope['second_deviation']['upper_bound']
    second_lower = raw_envelope['second_deviation']['lower_bound']
    for i in range(io_width):
        dim = {}
        dim['variable'] = headers[i]
        dim['first_deviation'] = {}
        dim['second_deviation'] = {}
        dim['mean'] = mean[:, i].tolist()
        dim['standard_deviation'] = sd[:, i].tolist()
        dim['first_deviation']['upper_bound'] = first_upper[:, i].tolist()
        dim['first_deviation']['lower_bound'] = first_lower[:, i].tolist()
        dim['second_deviation']['upper_bound'] = second_upper[:, i].tolist()
        dim['second_deviation']['lower_bound'] = second_lower[:, i].tolist()
        formatted_envelope.append(dim)
    return formatted_envelope
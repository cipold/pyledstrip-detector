import numpy as np


def lin(a_pos, a_val, b_pos, b_val, pos):
    diff = b_pos - a_pos
    if abs(diff) == 0:
        return a_val

    return (a_val - b_val) * (b_pos - pos) / diff + b_val


def convert_map(led_map):
    return [[x, y] for x, y in led_map.values()]


def complete_map(led_map, led_count):
    complete_led_map = {}

    last = next(iter(led_map.items()))
    for entry in led_map.items():
        for i in range(last[0], entry[0]):
            complete_led_map[i] = (
                lin(last[0], last[1][0], entry[0], entry[1][0], i),
                lin(last[0], last[1][1], entry[0], entry[1][1], i)
            )

        last = entry

    for i in range(last[0], led_count):
        complete_led_map[i] = (last[1][0], last[1][1])

    return complete_led_map


def median_filter(v):
    return [v[0]] + [np.median([a, b, c]) for a, b, c in zip(v[:-2], v[1:-1], v[2:])] + [v[-1]]


def filter_map(led_map):
    x_values = [x for (x, _) in led_map.values()]
    y_values = [y for (_, y) in led_map.values()]

    x_values = median_filter(x_values)
    y_values = median_filter(y_values)

    return {k: (x, y) for k, x, y in zip(led_map.keys(), x_values, y_values)}


def gaussian_filter(v):
    v = list(np.convolve([v[0]] + v + [v[-1]], [0.25, 0.5, 0.25], 'valid'))
    v = list(np.convolve([v[0]] + v + [v[-1]], [0.25, 0.5, 0.25], 'valid'))
    return v


def filter_complete_map(led_map):
    x_values = [x for (x, _) in led_map.values()]
    y_values = [y for (_, y) in led_map.values()]

    x_values = gaussian_filter(x_values)
    y_values = gaussian_filter(y_values)

    return {k: (x, y) for k, x, y in zip(led_map.keys(), x_values, y_values)}

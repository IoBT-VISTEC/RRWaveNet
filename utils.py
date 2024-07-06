from scipy.signal import resample
import numpy as np


def resampling(inp, window_size, sampling_rate=50):
    """
    Return an array with the same structure but the data is now sampled with the chosen rate.
    """
    samples_per_window = sampling_rate * window_size
    resampled = [[resample(window, samples_per_window)
                  for window in patient] for patient in inp]
    return resampled


def check_dimension(inp, window_size, sampling_rate=50):
    """
    Boolean. Check whether the array is correctly preprocessed.
    """
    samples_per_window = sampling_rate * window_size
    return all(window.shape == (samples_per_window, ) for patient in inp for window in patient)


def stack(inp, mode="vstack"):
    """
    Stack the array to make the dimension to 3D (# windows, # data points, 1)
    where # data points = samples_per_window. The input should be resampled beforehand.
    """
    _stack = getattr(np, mode)

    if mode == "vstack":
        return _stack(inp).reshape(-1, np.shape(inp[0][0])[0], 1)

    if mode == "hstack":
        return _stack(inp).reshape(-1, 1)

    raise KeyError

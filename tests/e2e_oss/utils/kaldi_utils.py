"""Kaldi tests utilities."""
from collections import OrderedDict
import numpy as np
try:
    from kaldiio import load_ark
except ImportError:
    pass


def read_ark_data(ark_path, utterances_to_read=None, frames_per_utterance=None):
    """Reads ark file & filters its content if utterances_to_read or frames_per_utterance are not None
    :param ark_path: path to .ark file to read
    :param utterances_to_read: list of utterance names to read from input .ark file. If None, read all utterances
    :param frames_per_utterance: number of frames to read from each utterance. If None, read all frames
    :return: Dictionary with ark file data
    """
    # load_ark returns generator object
    ark_data = OrderedDict({k: v for k, v in load_ark(ark_path)})

    if utterances_to_read and not any(utt in ark_data.keys() for utt in utterances_to_read):
        raise ValueError(
            "Some of utterance names specified in utterances_to_read: {} are missing in input {}: {}"
            .format(utterances_to_read, ark_path, ark_data.keys()))

    filtered_ark_data = {}
    for utterance_key, frames_data in ark_data.items():
        if frames_per_utterance:
            frames_data = frames_data[:frames_per_utterance, :]
        if not utterances_to_read or utterance_key in utterances_to_read:
            filtered_ark_data.update({utterance_key: frames_data})

    return filtered_ark_data


def get_quantization_scale_factors(input_data):
    target_max = 16384
    data = OrderedDict()
    for frame in input_data:
        for input in frame:
            frame_data = frame[input]
            data[input] = frame_data if input not in data else np.concatenate((data[input], frame_data))
    scale_factors = OrderedDict()
    for input_name in data:
        abs_max = abs(max(data[input_name].max(), data[input_name].min(), key=abs))
        scale_factor = 1 if abs_max == 0 else target_max / abs_max
        scale_factors[input_name] = scale_factor
    return scale_factors

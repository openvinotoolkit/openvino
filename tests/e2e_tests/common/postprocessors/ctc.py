# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CTC output postprocessor"""
import string

import numpy as np
from tensorflow.keras import backend as k

from .provider import ClassProvider


class ParseCTCOutput(ClassProvider):
    """Transforms CTC output to dictionary {"predictions": predicted_strings, ""probs": corresponding_probabilities}"""
    __action_name__ = "ctc_decode"

    def __init__(self, config):
        self.top_paths = config.get("top_paths")
        self.beam_width = config.get("beam_width")

    def ctc_decode(self, data):
        """
        Parse CTC output
        Source:
        https://intel-my.sharepoint.com/:u:/r/personal/abdulmecit_gungor_intel_com/Documents/Perpetuuiti/OCR-HandWritten/src/network/model.py?csf=1&web=1&e=ZGZ8nO
        """
        predicts, probabilities = [], []
        input_length = len(max(data, key=len))
        data_len = np.asarray([input_length for _ in range(len(data))])
        decode, logs = k.ctc_decode(data, data_len, greedy=False, beam_width=self.beam_width, top_paths=self.top_paths)
        probabilities.extend([np.exp(x) for x in logs])
        decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
        predicts.extend(np.swapaxes(decode, 0, 1))

        return predicts, probabilities

    @staticmethod
    def to_text(text, chars):
        """Decode vector to text"""
        pad_tk, unk_tk = "¶", "¤"
        chars = pad_tk + unk_tk + chars
        decoded = "".join([chars[int(char)] for char in text if char > -1])
        return decoded

    def apply(self, data: dict):
        predicts, probs = [], []
        assert len(data.keys()) == 1, \
            "Expected 1 output layer, but got {} layers".format(len(data.keys()))

        layer = iter(data.keys())
        data = data[next(layer)]
        for b in range(len(data)):
            cur_predicts, cur_probs = self.ctc_decode(data)
            charset_base = string.printable[:95]
            cur_predicts = [[self.to_text(x, charset_base) for x in y] for y in cur_predicts]
            predicts.extend(cur_predicts)
            probs.extend(cur_probs)
        decoded_output = {"predictions": predicts, "probs": probs}
        return decoded_output

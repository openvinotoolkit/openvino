# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import onnx

from openvino.tools.mo.front.onnx.lstm_ext import LSTMFrontExtractor
from unit_tests.utils.extractors import PB


class LSTMExtractorTest(unittest.TestCase):
    @staticmethod
    def _create_node(**attrs):
        pb = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R', 'B',],
            outputs=['Y', 'Y_h', 'Y_c'],
            hidden_size=128,
            **attrs,
        )
        node = PB({'pb': pb})
        return node

    base_attrs = {
        'type': 'RNNSequence',
        'op': 'LSTM',
        'batch_dim': 1,
        'sequence_dim': 0,
        'blobs_wrb': True,
        'has_num_directions': True,
        'num_layers': 1,
        'format': 'onnx',
        'multilayers': False,
        'gate_order': np.array([2, 0, 3, 1]),
        'direction': 'forward',
    }

    def test_base_attrs(self):
        node = self._create_node()
        LSTMFrontExtractor.extract(node)

        exp_res = self.base_attrs

        for key in exp_res.keys():
            equal = np.all(np.equal(node[key], exp_res[key], dtype=object))
            self.assertTrue(equal)

    def test_additional_attributes(self):
        additional_attrs = {
            'activation_alpha': [1.0, 0.0, 2.0],
            'activations': [b'relu', b'tanh', b'sigmoid'],
            'clip': 10.0,
        }

        node = self._create_node(**additional_attrs)
        LSTMFrontExtractor.extract(node)

        exp_res = dict(**self.base_attrs, **additional_attrs)
        exp_res['activations'] = ['relu', 'tanh', 'sigmoid']

        for key in exp_res.keys():
            equal = np.all(np.equal(node[key], exp_res[key], dtype=object))
            self.assertTrue(equal, 'Values for attr {} are not equal'.format(key))

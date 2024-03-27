# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.mxnet.RNN_ext import RNNFrontExtractor
from openvino.tools.mo.utils.error import Error
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from unit_tests.utils.extractors import PB


class RNNFrontExtractorTest(UnitTestWithMockedTelemetry):
    @staticmethod
    def _create_node(**attrs):
        params = {'attrs': {
            **attrs
        }}
        node = PB({'symbol_dict': params})
        return node

    base_attrs = {
        'batch_dim': 1,
        'sequence_dim': 0,
        'blobs_wrb': False,
        'format': 'mxnet',
        'gate_order': [1, 0, 2, 3],
    }

    def test_base_attrs(self):
        attrs = {
            'state_size': 128,
            'mode': 'lstm',
        }

        additional_attrs = {
            'multilayers': False,
            'hidden_size': 128,
            'has_num_directions': False,
            'direction': 'forward',
            'num_layers': 1,
        }

        node = self._create_node(**attrs)
        RNNFrontExtractor.extract(node)

        expect_attrs = {**self.base_attrs, **additional_attrs}

        for key in expect_attrs.keys():
            equal = np.all(np.equal(node[key], expect_attrs[key], dtype=object))
            self.assertTrue(equal, 'Values for attr {} are not equal'.format(key))

        self.assertTrue(node.op == 'LSTM')

    def test_unsupported_mode(self):
        attrs = {
            'state_size': 128,
            'mode': 'abracadabra',
        }
        node = self._create_node(**attrs)
        with self.assertRaises(Error):
            RNNFrontExtractor.extract(node)

    def test_additional_attrs(self):
        attrs = {
            'state_size': 128,
            'mode': 'lstm',
            'bidirectional': True,
            'num_layers': 2,
        }

        additional_attrs = {
            'multilayers': True,
            'hidden_size': 128,
            'has_num_directions': True,
            'direction': 'bidirectional',
            'num_layers': 2,
        }

        node = self._create_node(**attrs)
        RNNFrontExtractor.extract(node)

        expect_attrs = {**self.base_attrs, **additional_attrs}

        for key in expect_attrs.keys():
            equal = np.all(np.equal(node[key], expect_attrs[key], dtype=object))
            self.assertTrue(equal, 'Values for attr {} are not equal'.format(key))

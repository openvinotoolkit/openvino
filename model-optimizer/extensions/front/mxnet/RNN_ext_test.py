"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import unittest

import numpy as np

from extensions.front.mxnet.RNN_ext import RNNFrontExtractor
from mo.utils.error import Error
from mo.utils.unittest.extractors import PB


class RNNFrontExtractorTest(unittest.TestCase):
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
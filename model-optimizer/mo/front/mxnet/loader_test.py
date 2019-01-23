"""
 Copyright (c) 2018 Intel Corporation

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
from unittest.mock import patch

from mo.front.mxnet.loader import load_symbol_nodes, parse_input_model


class MockSymbolLoadObj():
    def tojson(self):
        pass


class TestLoader(unittest.TestCase):
    @patch('json.load')
    @patch('json.loads')
    @patch('os.path.isfile')
    @patch('mxnet.symbol.load')
    def test_load_symbol_nodes(self, mock_symbol_load, mock_isfile, mock_json_loads, mock_json_load):
        mock_isfile.return_value = True
        mock_json_load.return_value = {'nodes': ''}
        mock_json_loads.return_value = {'nodes': {'node1': 1}}
        mock_symbol_load_obj = MockSymbolLoadObj()
        mock_symbol_load.return_value = mock_symbol_load_obj
        with patch('mo.front.mxnet.loader.open') as mock_open:
            self.assertEqual({'node1': 1}, load_symbol_nodes("model_name", True))

    @patch('json.load')
    @patch('json.loads')
    @patch('os.path.isfile')
    @patch('mxnet.symbol.load')
    def test_load_symbol_with_custom_nodes(self, mock_symbol_load, mock_isfile, mock_json_loads, mock_json_load):
        mock_isfile.return_value = True
        mock_json_load.return_value = {'nodes': [{'op': 'custom_op'}, {'op': 'custom_op'}]}
        mock_json_loads.return_value = {'nodes': {'node1': 1}}
        mock_symbol_load_obj = MockSymbolLoadObj()
        mock_symbol_load.return_value = mock_symbol_load_obj
        with patch('mo.front.mxnet.loader.open') as mock_open:
            list_nodes = load_symbol_nodes("model_name", False)
            self.assertEqual(2, len(list_nodes))
            for node in list_nodes:
                self.assertEqual({'op': 'custom_op'}, node)

    def test_parse_input_model(self):
        input_model = '/model-optimizer-mxnet/data/nd/vgg19-0015.params'
        model_name, iteration_number = parse_input_model(input_model)
        self.assertEqual(model_name, '/model-optimizer-mxnet/data/nd/vgg19')
        self.assertEqual(iteration_number, 15)

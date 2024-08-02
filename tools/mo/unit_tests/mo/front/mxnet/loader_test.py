# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from openvino.tools.mo.front.mxnet.loader import load_symbol_nodes, parse_input_model
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry


class MockSymbolLoadObj():
    def tojson(self):
        pass


class TestLoader(UnitTestWithMockedTelemetry):
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
        with patch('openvino.tools.mo.front.mxnet.loader.open') as mock_open:
            self.assertEqual({'node1': 1}, load_symbol_nodes("model_name", legacy_mxnet_model=True))

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
        with patch('openvino.tools.mo.front.mxnet.loader.open') as mock_open:
            list_nodes = load_symbol_nodes("model_name", legacy_mxnet_model=False)
            self.assertEqual(2, len(list_nodes))
            for node in list_nodes:
                self.assertEqual({'op': 'custom_op'}, node)

    def test_parse_input_model(self):
        input_model = '/model-optimizer-mxnet/data/nd/vgg19-0015.params'
        model_name, iteration_number = parse_input_model(input_model)
        self.assertEqual(model_name, '/model-optimizer-mxnet/data/nd/vgg19')
        self.assertEqual(iteration_number, 15)


    @patch('json.load')
    @patch('json.loads')
    @patch('os.path.isfile')
    @patch('mxnet.symbol.load')
    def test_load_symbol_nodes_with_json_and_lagacy_mode(self, mock_symbol_load, mock_isfile, mock_json_loads, mock_json_load):
        mock_isfile.return_value = True
        mock_json_load.return_value = {'nodes': ''}
        mock_json_loads.return_value = {'nodes': {'node1': 1}}
        mock_symbol_load_obj = MockSymbolLoadObj()
        mock_symbol_load.return_value = mock_symbol_load_obj
        with patch('openvino.tools.mo.front.mxnet.loader.open') as mock_open:
            self.assertEqual({'node1': 1}, load_symbol_nodes("model_name", input_symbol="some-symbol.json", legacy_mxnet_model=True))


    @patch('json.load')
    @patch('json.loads')
    @patch('os.path.isfile')
    @patch('mxnet.symbol.load')
    def test_load_symbol_nodes_with_json(self, mock_symbol_load, mock_isfile, mock_json_loads, mock_json_load):
        mock_isfile.return_value = True
        #json.load
        mock_json_load.return_value = {'nodes': {'node1': 1}}
        mock_json_loads.return_value = {'nodes': ''}
        mock_symbol_load_obj = MockSymbolLoadObj()
        mock_symbol_load.return_value = mock_symbol_load_obj
        with patch('openvino.tools.mo.front.mxnet.loader.open') as mock_open:
            self.assertEqual({'node1': 1}, load_symbol_nodes("model_name", input_symbol="some-symbol.json", legacy_mxnet_model=False))

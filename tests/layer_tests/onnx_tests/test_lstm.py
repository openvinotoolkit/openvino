# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestLSTM(OnnxRuntimeLayerTest):
    skip_framework = True

    def create_lstm(self, direction: str, cell_type: str, hidden_size=128):
        """
            ONNX net

            Input->LSTM->Output   =>   Only accuracy check

        """

        #   Create ONNX model

        import onnx
        from onnx import helper
        from onnx import TensorProto

        assert cell_type in ['LSTM', 'RNN', 'GRU']
        assert direction in ['forward', 'reverse', 'bidirectional']
        n_gates = {'LSTM': 4, 'RNN': 1, 'GRU': 3}
        M = n_gates[cell_type]

        seq_len = 10
        batch_size = 4
        input_size = 64
        num_direction = 1 if direction in ["forward", "reverse"] else 2

        input_shape = [seq_len, batch_size, input_size]
        output_shape = [seq_len, num_direction, batch_size, hidden_size]

        w_shape = [num_direction, M * hidden_size, input_size]
        r_shape = [num_direction, M * hidden_size, hidden_size]

        init_h_shape = [num_direction, batch_size, hidden_size]
        init_c_shape = [num_direction, batch_size, hidden_size]

        init_h_value = np.ones(init_h_shape, dtype=np.float32)
        init_c_value = np.ones(init_c_shape, dtype=np.float32)

        w_value = np.ones(w_shape, dtype=np.float32)
        r_value = np.ones(r_shape, dtype=np.float32)

        # Creating LSTM Operation
        x = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
        y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

        w = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['W'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=w_value.shape,
                vals=w_value.flatten().astype(float),
            ),
        )

        r = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['R'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=r_value.shape,
                vals=r_value.flatten().astype(float),
            ),
        )

        init_h = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['init_h'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=init_h_value.shape,
                vals=init_h_value.flatten().astype(float),
            ),
        )

        inputs = ['X', 'W', 'R', '', '', 'init_h']

        if cell_type == 'LSTM':
            init_c = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['init_c'],
                value=onnx.helper.make_tensor(
                    name='const_tensor',
                    data_type=onnx.TensorProto.FLOAT,
                    dims=init_c_value.shape,
                    vals=init_c_value.flatten().astype(float),
                ),
            )

            inputs.append('init_c')

        node_lstm = onnx.helper.make_node(
            cell_type,
            inputs=inputs,
            outputs=['', 'Y'],
            hidden_size=hidden_size,
            direction=direction,
        )

        # Create the graph (GraphProto)
        if cell_type == 'LSTM':
            graph_def = helper.make_graph(
                [w, r, init_h, init_c, node_lstm],
                'test_lstm',
                [x],
                [y],
            )
        else:
            graph_def = helper.make_graph(
                [w, r, init_h, node_lstm],
                'test_lstm',
                [x],
                [y],
            )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_{}_model'.format(cell_type))

        # We do not create reference graph, as it's too complicated to construct it
        # Moreover, IR reader do not support TensorIterator layers
        # So we return None to skip IR comparision

        return onnx_net, None

    @pytest.mark.precommit
    @pytest.mark.timeout(250)
    @pytest.mark.parametrize('direction', ["forward", "bidirectional", "reverse"])
    @pytest.mark.parametrize('cell_type', ["LSTM", "GRU", "RNN"])
    def test_lstm_simple_precommit(self, direction, cell_type, ie_device, precision, ir_version,
                                   temp_dir):
        self._test(*self.create_lstm(direction, cell_type), ie_device, precision, ir_version,
                   temp_dir=temp_dir, infer_timeout=150)

    # LSTM/RNN/GRU Sequence Generation
    @pytest.mark.parametrize('direction', ["forward", "bidirectional", "reverse"])
    @pytest.mark.parametrize('cell_type', ["LSTM", "GRU", "RNN"])
    def test_lstm_sequence_generate(self, direction, cell_type, ie_device, precision, ir_version,
                                    temp_dir):
        self._test(*self.create_lstm(direction, cell_type), ie_device, precision, ir_version,
                   disabled_transforms='lstm_to_tensor_iterator,gru_and_rnn_to_tensor_iterator',
                   temp_dir=temp_dir)

    # TODO: add more params for nightly
    @pytest.mark.nightly
    @pytest.mark.parametrize('direction', ["forward", "bidirectional", "reverse"])
    @pytest.mark.parametrize('cell_type', ["LSTM", "GRU", "RNN"])
    def test_lstm_nightly(self, direction, cell_type, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_lstm(direction, cell_type), ie_device, precision, ir_version,
                   temp_dir=temp_dir)

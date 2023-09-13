# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest


class TestScale(OnnxRuntimeLayerTest):
    def create_net(self, dtype, shape, mean, scale, seed, ir_version):
        """
            ONNX net                     IR net

            RandomNormal->Output   =>   RandomNormal->Output

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        output = helper.make_tensor_value_info('output', dtype, shape)

        node_def = onnx.helper.make_node(
            'RandomNormal',
            inputs=[],
            outputs=['output'],
            dtype=dtype,
            mean=mean,
            scale=scale,
            seed=seed,
            shape=shape,
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = helper.make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #

        ref_net = None

        return onnx_net, ref_net

    from onnx import TensorProto
    test_data = [dict(dtype=TensorProto.FLOAT, shape=[10, 12], mean=0.0, scale=1.0, seed=2453.57),]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_scale(self, params, ie_device, precision, ir_version, temp_dir, use_old_api):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir, use_old_api=use_old_api)


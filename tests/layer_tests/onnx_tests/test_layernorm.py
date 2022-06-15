# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest

from unit_tests.utils.graph import build_graph


class TestLN(OnnxRuntimeLayerTest):
    def create_net(self, shape, epsilon, axis, stash_type, ir_version):
        """
            ONNX net                   IR net

            Input->LN->Output   =>    Input->Norm->Power

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)
     
        other_shape = shape.copy()[axis:]

        min_val = -127
        const1 = np.random.randint(min_val, 127, other_shape).astype(float)
        const2 = np.random.randint(min_val, 127, other_shape).astype(float)
        print('scale={}'.format(const1))
        print('bias={}'.format(const2))
        node_scale_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['scale'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=const1.shape,
                vals=const1.flatten(),
            ),
        )

        node_bias_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['bias'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=const2.shape,
                vals=const2.flatten(),
            ),
        )

        args = dict()
        args['epsilon'] = epsilon
        args['axis'] = axis
        if stash_type != None :
            args['stash_type'] = stash_type
        
        node_def = onnx.helper.make_node(
            'LayerNormalization',
            inputs=['input', 'scale', 'bias'],
            outputs=['output'],
            **args
        )
        
        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_scale_def, node_bias_def, node_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = helper.make_model(graph_def, producer_name='test_model')

        #   Create reference IR net
        ref_net = None

        return onnx_net, ref_net


    test_data = [
        dict(shape=[3], epsilon=0.002, axis=0, stash_type=1),
        dict(shape=[1, 3], epsilon=0.002, axis=0, stash_type=1),
        dict(shape=[1, 3], epsilon=0.002, axis=1, stash_type=1),
        dict(shape=[1, 2, 3], epsilon=0.002, axis=0, stash_type=None),
        dict(shape=[1, 2, 3], epsilon=0.002, axis=1, stash_type=None),
        dict(shape=[1, 2, 3], epsilon=0.002, axis=2, stash_type=None),
        dict(shape=[1, 2, 3, 4], epsilon=0.002, axis=0, stash_type=None),
        dict(shape=[1, 2, 3, 4], epsilon=0.002, axis=1, stash_type=None),
        dict(shape=[1, 2, 3, 4], epsilon=0.002, axis=2, stash_type=None),
        dict(shape=[1, 2, 3, 4], epsilon=0.002, axis=3, stash_type=None),
        #dict(shape=[2, 3, 10, 12], shape2=[2, 3, 10, 12], epsilon=0.0002),
        #dict(shape=[2, 3, 8, 10, 12], shape2=[2, 3, 8, 10, 12], epsilon=0.0002)
        ]



    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_ln(self, params, ie_device, precision, ir_version, temp_dir, api_2):
 #       self.skip_framework = True
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir, api_2=api_2)


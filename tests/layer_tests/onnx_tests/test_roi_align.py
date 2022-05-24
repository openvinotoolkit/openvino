# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest

from unit_tests.utils.graph import build_graph


class TestROIAlign(OnnxRuntimeLayerTest):
    def create_net(self, input_shape, rois_shape, indices_shape, output_shape,
                   pooled_h, pooled_w, mode, sampling_ratio, spatial_scale, ir_version):
        """
            ONNX net                    IR net

            Input->ROIAlign->Output   =>    Parameter->ROIAlign->Result

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        rois = helper.make_tensor_value_info('rois', TensorProto.FLOAT, rois_shape)
        indices = helper.make_tensor_value_info('indices', TensorProto.INT64, indices_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        node_def = onnx.helper.make_node(
            'ROIAlign',
            inputs=['input', 'rois', 'indices'],
            outputs=['output'],
            **{'output_height': pooled_h, 'output_width': pooled_w, 'mode': mode,
               'sampling_ratio': sampling_ratio, 'spatial_scale': spatial_scale},
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [input, rois, indices],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = helper.make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #

        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                # comparison in these tests starts from input node, as we have 3 of them IREngine gets confused
                # and takes the first input node in inputs list sorted by lexicographical order
                '1_input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': input_shape, 'kind': 'data'},

                '2_rois': {'kind': 'op', 'type': 'Parameter'},
                'rois_data': {'shape': rois_shape, 'kind': 'data'},

                '3_indices': {'kind': 'op', 'type': 'Parameter'},
                'indices_data': {'shape': indices_shape, 'kind': 'data'},

                'node': {'kind': 'op', 'type': 'ROIAlign', 'pooled_h': pooled_h,
                         'pooled_w': pooled_w,
                         'mode': mode, 'sampling_ratio': sampling_ratio,
                         'spatial_scale': spatial_scale},
                'node_data': {'shape': output_shape, 'kind': 'data'},

                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [
                                      ('1_input', 'input_data'),
                                      ('input_data', 'node', {'in': 0}),
                                      ('2_rois', 'rois_data'),
                                      ('rois_data', 'node', {'in': 1}),
                                      ('3_indices', 'indices_data'),
                                      ('indices_data', 'node', {'in': 2}),

                                      ('node', 'node_data'),
                                      ('node_data', 'result')
                                  ])
        return onnx_net, ref_net

    test_data = [
        dict(input_shape=[1, 256, 200, 272], rois_shape=[1000, 4], indices_shape=[1000],
             pooled_h=7, pooled_w=7, mode="avg", sampling_ratio=2, spatial_scale=0.25,
             output_shape=[1000, 256, 7, 7]),
        dict(input_shape=[7, 256, 200, 200], rois_shape=[1000, 4], indices_shape=[1000],
             pooled_h=6, pooled_w=6, mode="max", sampling_ratio=2, spatial_scale=16.0,
             output_shape=[1000, 256, 6, 6]),
        dict(input_shape=[7, 256, 200, 200], rois_shape=[1000, 4], indices_shape=[1000],
             pooled_h=5, pooled_w=6, mode="max", sampling_ratio=2, spatial_scale=16.0,
             output_shape=[1000, 256, 5, 6]),

    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_roi_align(self, params, ie_device, precision, ir_version, temp_dir, api_2):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir, api_2=api_2)

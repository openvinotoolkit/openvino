# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model
from unit_tests.utils.graph import build_graph


class TestROIAlign(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            if input == 'indices':
                if isinstance(inputs_dict['input'], list):
                    batch = inputs_dict['input'][0]
                else:
                    batch = inputs_dict['input'].shape[0]
                inputs_dict[input] = np.random.choice(range(batch), inputs_dict[input])
            elif input == 'input':
                inputs_dict[input] = np.ones(inputs_dict[input]).astype(np.float32)
            else:
                inputs_dict[input] = np.random.randint(-255, 255, inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_net(self, input_shape, rois_shape, indices_shape, output_shape,
                   pooled_h, pooled_w, mode, sampling_ratio, spatial_scale, ir_version, onnx_version):
        """
            ONNX net                    IR net

            Input->ROIAlign->Output   =>    Parameter->ROIAlign->Result

        """


        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto, OperatorSetIdProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        rois = helper.make_tensor_value_info('rois', TensorProto.FLOAT, rois_shape)
        indices = helper.make_tensor_value_info('indices', TensorProto.INT64, indices_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        node_def = onnx.helper.make_node(
            'RoiAlign',
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

        operatorsetid = OperatorSetIdProto()
        operatorsetid.domain = ""
        operatorsetid.version = onnx_version
        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model', opset_imports=[operatorsetid])

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
        dict(input_shape=[1, 90, 12, 14], rois_shape=[5, 4], indices_shape=[5],
             pooled_h=2, pooled_w=2, mode="avg", sampling_ratio=2, spatial_scale=0.25,
             output_shape=[5, 90, 2, 2]),
        dict(input_shape=[1, 20, 12, 14], rois_shape=[5, 4], indices_shape=[5],
             pooled_h=2, pooled_w=2, mode="avg", sampling_ratio=2, spatial_scale=0.25,
             output_shape=[5, 20, 2, 2]),
        dict(input_shape=[1, 50, 12, 14], rois_shape=[5, 4], indices_shape=[5],
             pooled_h=2, pooled_w=2, mode="avg", sampling_ratio=2, spatial_scale=0.25,
             output_shape=[5, 50, 2, 2]),
        dict(input_shape=[1, 120, 12, 14], rois_shape=[5, 4], indices_shape=[5],
             pooled_h=2, pooled_w=2, mode="avg", sampling_ratio=2, spatial_scale=0.25,
             output_shape=[5, 120, 2, 2]),
        dict(input_shape=[7, 1, 4, 4], rois_shape=[2, 4], indices_shape=[2],
             pooled_h=2, pooled_w=2, mode="max", sampling_ratio=2, spatial_scale=16.0,
             output_shape=[2, 1, 2, 2]),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Windows', reason="Ticket - 122731")
    @pytest.mark.xfail(condition=platform.system() in ('Linux', 'Darwin') and platform.machine() in ('arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'),
                       reason='Ticket - 122846, 122783, 126312')
    def test_roi_alignv10(self, params, ie_device, precision, ir_version, temp_dir):
        # TODO: ticket for investigating GPU failures: CVS-86300
        if ie_device != "GPU":
            self._test(*self.create_net(**params, ir_version=ir_version, onnx_version=10), ie_device, precision,
                       ir_version,
                       temp_dir=temp_dir,
                       use_legacy_frontend=True)

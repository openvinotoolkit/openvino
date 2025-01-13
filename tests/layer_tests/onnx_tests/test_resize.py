# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_np, \
    np_data_type_to_destination_type
from unit_tests.utils.graph import build_graph


class TestResize(OnnxRuntimeLayerTest):
    def create_resize_net(self, input_shape, output_shape, scales, sizes,
                          coordinate_transformation_mode, cubic_coeff_a, mode,
                          nearest_mode, precision, ir_version):
        import onnx
        from onnx import helper
        from onnx import TensorProto

        input_rank = len(input_shape)

        roi_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['roi'],
            value=helper.make_tensor(
                name='roi_consts',
                data_type=TensorProto.FLOAT,
                dims=[2 * input_rank],
                vals=np.array([*np.zeros(input_rank), *np.ones(input_rank)])
            )
        )

        onnx_scales = scales
        if scales is None:
            onnx_scales = np.array(output_shape).astype(float) / np.array(input_shape).astype(
                float)
        scales_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['scales'],
            value=helper.make_tensor(
                name='scales_const',
                data_type=TensorProto.FLOAT,
                dims=[len(output_shape)],
                vals=onnx_scales
            )
        )

        nodes_list = [roi_node, scales_node]
        inputs_list = ['input', 'roi', 'scales']

        if sizes is not None:
            sizes_node = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['sizes'],
                value=helper.make_tensor(
                    name='sizes_const',
                    data_type=TensorProto.INT64,
                    dims=[len(output_shape)],
                    vals=sizes
                )
            )

            nodes_list.append(sizes_node)
            inputs_list.append('sizes')

        args = dict()

        onnx_mode = mode or 'nearest'
        onnx_nearest_mode = nearest_mode or 'round_prefer_floor'
        cube_coeff = -0.75 if cubic_coeff_a is None else cubic_coeff_a
        onnx_coordinate_transformation_mode = coordinate_transformation_mode or 'half_pixel'

        args['nearest_mode'] = onnx_nearest_mode
        args['mode'] = onnx_mode
        args['cubic_coeff_a'] = cube_coeff
        args['coordinate_transformation_mode'] = onnx_coordinate_transformation_mode

        x = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        y = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        resize_node = onnx.helper.make_node(
            'Resize',
            inputs=inputs_list,
            outputs=['output'],
            **args,
        )

        nodes_list.append(resize_node)

        graph_def = onnx.helper.make_graph(nodes_list, 'test_model', [x], [y])

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')
        onnx.checker.check_model(onnx_net)

        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            if sizes is None and scales is None:
                return onnx_net, ref_net

            input_shape_as_array = int64_array(input_shape)

            if sizes is not None and scales is not None:
                shape_calculation_mode = 'sizes'
                sizes_value = int64_array(sizes)
                scales_value = np.array(scales).astype(float)
            elif sizes is not None and scales is None:
                shape_calculation_mode = 'sizes'
                sizes_value = int64_array(sizes)
                scales_value = sizes_value / input_shape_as_array
            else:
                shape_calculation_mode = 'scales'
                scales_value = np.array(scales).astype(float)
                sizes_value = np.floor(input_shape_as_array * scales_value + 1e-5).astype(np.int64)

            if precision == 'FP16':
                sizes_value = sizes_value.astype(np.float16)
                scales_value = scales_value.astype(np.float16)

            interp_mode = convert_onnx_mode(onnx_mode)

            interp_attrs = {
                'type': 'Interpolate',
                'kind': 'op',
                'mode': interp_mode,
                'shape_calculation_mode': shape_calculation_mode,
                'coordinate_transformation_mode': onnx_coordinate_transformation_mode,
                'nearest_mode': onnx_nearest_mode,
                'antialias': 0,
                'cube_coeff': cube_coeff,
                'pads_begin': np.zeros(input_rank).astype(np.int64),
                'pads_end': np.zeros(input_rank).astype(np.int64),
                'version': 'opset4'
            }

            if shape_calculation_mode == 'scales':
                ref_net = create_ref_net_in_scales_mode(precision, input_shape_as_array,
                                                        output_shape,
                                                        sizes_value, scales_value, interp_attrs)
            else:
                ref_net = create_ref_net_in_sizes_mode(precision, input_shape_as_array,
                                                       output_shape,
                                                       sizes_value, scales_value, interp_attrs)

        return onnx_net, ref_net

    test_data = [
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 3, 3],
             scales=[1.0, 1.0, 0.8, 0.8], sizes=None,
             coordinate_transformation_mode='half_pixel',
             cubic_coeff_a=None, mode='cubic', nearest_mode=None),
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 3, 3],
             scales=[1.0, 1.0, 0.8, 0.8], sizes=None,
             coordinate_transformation_mode='align_corners',
             cubic_coeff_a=None, mode='cubic', nearest_mode=None),
        dict(input_shape=[1, 1, 2, 4], output_shape=[1, 1, 1, 2],
             scales=[1.0, 1.0, 0.6, 0.6], sizes=None,
             coordinate_transformation_mode=None,
             cubic_coeff_a=None, mode='linear', nearest_mode=None),
        dict(input_shape=[1, 1, 2, 4], output_shape=[1, 1, 1, 2],
             scales=[1.0, 1.0, 0.6, 0.6], sizes=None,
             coordinate_transformation_mode='align_corners',
             cubic_coeff_a=None, mode='linear', nearest_mode=None),
        dict(input_shape=[1, 1, 2, 4], output_shape=[1, 1, 1, 2],
             scales=[1.0, 1.0, 0.6, 0.6], sizes=None,
             coordinate_transformation_mode=None,
             cubic_coeff_a=None, mode='nearest', nearest_mode=None),
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 8, 8],
             scales=[1.0, 1.0, 2.0, 2.0], sizes=None,
             coordinate_transformation_mode=None,
             cubic_coeff_a=None, mode='cubic', nearest_mode=None),
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 8, 8],
             scales=[1.0, 1.0, 2.0, 2.0], sizes=None,
             coordinate_transformation_mode='align_corners',
             cubic_coeff_a=None, mode='cubic', nearest_mode=None),
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 8, 8],
             scales=[1.0, 1.0, 2.0, 2.0], sizes=None,
             coordinate_transformation_mode='asymmetric',
             cubic_coeff_a=None, mode='cubic', nearest_mode=None),
        dict(input_shape=[1, 1, 2, 2], output_shape=[1, 1, 4, 4],
             scales=[1.0, 1.0, 2.0, 2.0], sizes=None,
             coordinate_transformation_mode=None,
             cubic_coeff_a=None, mode='linear', nearest_mode=None),
        dict(input_shape=[1, 1, 2, 2], output_shape=[1, 1, 4, 4],
             scales=[1.0, 1.0, 2.0, 2.0], sizes=None,
             coordinate_transformation_mode='align_corners',
             cubic_coeff_a=None, mode='linear', nearest_mode=None),
        dict(input_shape=[1, 1, 2, 2], output_shape=[1, 1, 4, 4],
             scales=[1.0, 1.0, 2.0, 2.0], sizes=None,
             coordinate_transformation_mode=None,
             cubic_coeff_a=None, mode='nearest', nearest_mode=None)
    ]

    @pytest.mark.parametrize("params", test_data)
    def test_resize(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_resize_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, custom_eps=2.0e-4, temp_dir=temp_dir)

    test_data_cubic = [
        dict(input_shape=[1, 3, 100, 200], output_shape=[1, 3, 350, 150],
             scales=[1.0, 1.0, 3.5, 150 / 200], sizes=None),
        dict(input_shape=[16, 7, 190, 400], output_shape=[16, 7, 390, 600],
             scales=[1.0, 1.0, 390 / 190, 600 / 400], sizes=None),
        dict(input_shape=[4, 33, 1024, 800], output_shape=[4, 33, 512, 800],
             scales=[1.0, 1.0, 0.5, 1.0], sizes=None),
        dict(input_shape=[4, 33, 3, 800], output_shape=[4, 33, 1, 800],
             scales=[1.0, 1.0, 0.3333334, 1.0], sizes=None),
        dict(input_shape=[100, 200], output_shape=[350, 150],
             scales=[3.5, 150 / 200], sizes=None),
        dict(input_shape=[190, 400], output_shape=[390, 600],
             scales=[390 / 190, 600 / 400], sizes=None),
        dict(input_shape=[1024, 800], output_shape=[512, 800],
             scales=[0.5, 1.0], sizes=None),
        dict(input_shape=[3, 800], output_shape=[1, 800],
             scales=[0.3333334, 1.0], sizes=None)
    ]

    @pytest.mark.parametrize("params", test_data_cubic)
    @pytest.mark.parametrize("coordinate_transformation_mode",
                             ['half_pixel', 'pytorch_half_pixel', 'align_corners',
                              'asymmetric', 'tf_half_pixel_for_nn'])
    @pytest.mark.parametrize("cubic_coeff_a", [-0.75])
    @pytest.mark.parametrize("mode", ['cubic'])
    @pytest.mark.parametrize("nearest_mode", ['round_prefer_floor'])
    def test_resize_combined_cubic(self, params, coordinate_transformation_mode, cubic_coeff_a,
                                   mode,
                                   nearest_mode, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_resize_net(**params,
                                           coordinate_transformation_mode=coordinate_transformation_mode,
                                           cubic_coeff_a=cubic_coeff_a, mode=mode,
                                           nearest_mode=nearest_mode,
                                           precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, custom_eps=2.6e-2, temp_dir=temp_dir)

    test_data_nearest = [
        dict(input_shape=[1, 3, 100, 200], output_shape=[1, 3, 350, 150],
             scales=[1.0, 1.0, 3.5, 150 / 200], sizes=None),
        dict(input_shape=[16, 7, 190, 400], output_shape=[16, 7, 390, 600],
             scales=[1.0, 1.0, 390 / 190, 600 / 400], sizes=None),
        dict(input_shape=[4, 33, 600, 800], output_shape=[4, 33, 300, 800],
             scales=[1.0, 1.0, 0.5, 1.0], sizes=None),
        dict(input_shape=[4, 33, 3, 800], output_shape=[4, 33, 1, 800],
             scales=[1.0, 1.0, 0.3333334, 1.0], sizes=None),
    ]

    @pytest.mark.parametrize("params", test_data_nearest)
    @pytest.mark.parametrize("coordinate_transformation_mode",
                             ['half_pixel', 'pytorch_half_pixel', 'align_corners',
                              'asymmetric', 'tf_half_pixel_for_nn'])
    @pytest.mark.parametrize("cubic_coeff_a", [-0.75])
    @pytest.mark.parametrize("mode", ['nearest'])
    @pytest.mark.parametrize("nearest_mode", ['round_prefer_floor', 'round_prefer_ceil',
                                              'floor', 'ceil'])
    def test_resize_combined_nearest(self, params, coordinate_transformation_mode, cubic_coeff_a,
                                     mode,
                                     nearest_mode, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_resize_net(**params,
                                           coordinate_transformation_mode=coordinate_transformation_mode,
                                           cubic_coeff_a=cubic_coeff_a, mode=mode,
                                           nearest_mode=nearest_mode,
                                           precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_linear = [
        dict(input_shape=[1, 3, 100, 200], output_shape=[1, 3, 350, 150],
             scales=[1.0, 1.0, 3.5, 150 / 200], sizes=None),
        dict(input_shape=[16, 7, 190, 400], output_shape=[16, 7, 390, 600],
             scales=[1.0, 1.0, 390 / 190, 600 / 400], sizes=None),
        dict(input_shape=[4, 33, 600, 800], output_shape=[4, 33, 300, 800],
             scales=[1.0, 1.0, 0.5, 1.0], sizes=None),
        dict(input_shape=[4, 33, 3, 800], output_shape=[4, 33, 1, 800],
             scales=[1.0, 1.0, 0.3333334, 1.0], sizes=None),
        dict(input_shape=[100, 200], output_shape=[350, 150],
             scales=[3.5, 150 / 200], sizes=None),
        dict(input_shape=[190, 400], output_shape=[390, 600],
             scales=[390 / 190, 600 / 400], sizes=None),
        dict(input_shape=[600, 800], output_shape=[300, 800],
             scales=[0.5, 1.0], sizes=None),
        dict(input_shape=[3, 800], output_shape=[1, 800],
             scales=[0.3333334, 1.0], sizes=None),
    ]

    @pytest.mark.parametrize("params", test_data_linear)
    @pytest.mark.parametrize("coordinate_transformation_mode",
                             ['half_pixel', 'pytorch_half_pixel', 'align_corners',
                              'asymmetric', 'tf_half_pixel_for_nn'])
    @pytest.mark.parametrize("cubic_coeff_a", [-0.75])
    @pytest.mark.parametrize("mode", ['linear'])
    @pytest.mark.parametrize("nearest_mode", ['round_prefer_floor'])
    def test_resize_combined_linear(self, params, coordinate_transformation_mode, cubic_coeff_a,
                                    mode,
                                    nearest_mode, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_resize_net(**params,
                                           coordinate_transformation_mode=coordinate_transformation_mode,
                                           cubic_coeff_a=cubic_coeff_a, mode=mode,
                                           nearest_mode=nearest_mode,
                                           precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, custom_eps=2.0e-2, temp_dir=temp_dir)

    test_data_sizes = [
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 3, 3],
             scales=None, sizes=[1, 1, 3, 3],
             coordinate_transformation_mode=None,
             cubic_coeff_a=None, mode='cubic', nearest_mode=None),
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 3, 1],
             scales=None, sizes=[1, 1, 3, 1],
             coordinate_transformation_mode='pytorch_half_pixel',
             cubic_coeff_a=None, mode='linear', nearest_mode=None),
        dict(input_shape=[1, 1, 2, 4], output_shape=[1, 1, 1, 3],
             scales=None, sizes=[1, 1, 1, 3],
             coordinate_transformation_mode=None,
             cubic_coeff_a=None, mode='nearest', nearest_mode=None),
        dict(input_shape=[1, 1, 2, 4], output_shape=[1, 1, 1, 2],
             scales=None, sizes=[1, 1, 1, 2],
             coordinate_transformation_mode=None,
             cubic_coeff_a=None, mode='nearest', nearest_mode=None),
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 3, 2],
             scales=None, sizes=[1, 1, 3, 2],
             coordinate_transformation_mode='tf_half_pixel_for_nn',
             cubic_coeff_a=None, mode='nearest', nearest_mode=None),
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 9, 10],
             scales=None, sizes=[1, 1, 9, 10],
             coordinate_transformation_mode=None,
             cubic_coeff_a=None, mode='cubic', nearest_mode=None),
        dict(input_shape=[1, 1, 2, 2], output_shape=[1, 1, 7, 8],
             scales=None, sizes=[1, 1, 7, 8],
             coordinate_transformation_mode=None,
             cubic_coeff_a=None, mode='nearest', nearest_mode=None),
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 8, 8],
             scales=None, sizes=[1, 1, 8, 8],
             coordinate_transformation_mode='half_pixel',
             cubic_coeff_a=None, mode='nearest', nearest_mode='ceil'),
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 8, 8],
             scales=None, sizes=[1, 1, 8, 8],
             coordinate_transformation_mode='align_corners',
             cubic_coeff_a=None, mode='nearest', nearest_mode='floor'),
        dict(input_shape=[1, 1, 4, 4], output_shape=[1, 1, 8, 8],
             scales=None, sizes=[1, 1, 8, 8],
             coordinate_transformation_mode='asymmetric',
             cubic_coeff_a=None, mode='nearest', nearest_mode='round_prefer_ceil'),
    ]

    @pytest.mark.parametrize("params", test_data_sizes)
    def test_resize_sizes(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_resize_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_sizes_cubic = [
        dict(input_shape=[1, 3, 100, 200], output_shape=[1, 3, 350, 150],
             scales=None, sizes=[1, 3, 350, 150]),
        dict(input_shape=[16, 7, 190, 400], output_shape=[16, 7, 390, 600],
             scales=None, sizes=[16, 7, 390, 600]),
        dict(input_shape=[4, 15, 700, 800], output_shape=[4, 15, 350, 800],
             scales=None, sizes=[4, 15, 350, 800]),
        dict(input_shape=[4, 15, 3, 200], output_shape=[4, 15, 1, 200],
             scales=None, sizes=[4, 15, 1, 200]),
        dict(input_shape=[100, 200], output_shape=[350, 150],
             scales=None, sizes=[350, 150]),
        dict(input_shape=[190, 400], output_shape=[390, 600],
             scales=None, sizes=[390, 600]),
        dict(input_shape=[700, 800], output_shape=[350, 800],
             scales=None, sizes=[350, 800]),
        dict(input_shape=[3, 200], output_shape=[1, 200],
             scales=None, sizes=[1, 200]),
    ]

    @pytest.mark.parametrize("params", test_data_sizes_cubic)
    @pytest.mark.parametrize("coordinate_transformation_mode",
                             ['half_pixel', 'pytorch_half_pixel', 'align_corners',
                              'asymmetric', 'tf_half_pixel_for_nn'])
    @pytest.mark.parametrize("cubic_coeff_a", [-0.75])
    @pytest.mark.parametrize("mode", ['cubic'])
    @pytest.mark.parametrize("nearest_mode", ['round_prefer_floor'])
    def test_resize_combined_sizes_cubic(self, params, coordinate_transformation_mode,
                                         cubic_coeff_a, mode,
                                         nearest_mode, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_resize_net(**params,
                                           coordinate_transformation_mode=coordinate_transformation_mode,
                                           cubic_coeff_a=cubic_coeff_a, mode=mode,
                                           nearest_mode=nearest_mode,
                                           precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, custom_eps=2.6e-2, temp_dir=temp_dir)

    test_data_sizes_nearest = [
        dict(input_shape=[1, 3, 100, 200], output_shape=[1, 3, 350, 150],
             scales=None, sizes=[1, 3, 350, 150]),
        dict(input_shape=[16, 7, 190, 400], output_shape=[16, 7, 390, 600],
             scales=None, sizes=[16, 7, 390, 600]),
        dict(input_shape=[4, 33, 600, 800], output_shape=[4, 33, 300, 800],
             scales=None, sizes=[4, 33, 300, 800]),
        dict(input_shape=[4, 33, 3, 800], output_shape=[4, 33, 1, 800],
             scales=None, sizes=[4, 33, 1, 800]),
        dict(input_shape=[3, 100, 200], output_shape=[3, 350, 150],
             scales=None, sizes=[3, 350, 150]),
        dict(input_shape=[7, 190, 400], output_shape=[7, 390, 600],
             scales=None, sizes=[7, 390, 600]),
        dict(input_shape=[33, 600, 800], output_shape=[33, 300, 800],
             scales=None, sizes=[33, 300, 800]),
        dict(input_shape=[33, 3, 800], output_shape=[33, 1, 800],
             scales=None, sizes=[33, 1, 800]),
        dict(input_shape=[100, 200], output_shape=[350, 150],
             scales=None, sizes=[350, 150]),
        dict(input_shape=[190, 400], output_shape=[390, 600],
             scales=None, sizes=[390, 600]),
        dict(input_shape=[600, 800], output_shape=[300, 800],
             scales=None, sizes=[300, 800]),
        dict(input_shape=[3, 800], output_shape=[1, 800],
             scales=None, sizes=[1, 800]),
        dict(input_shape=[100], output_shape=[350],
             scales=None, sizes=[350]),
        dict(input_shape=[190], output_shape=[390],
             scales=None, sizes=[390]),
        dict(input_shape=[600], output_shape=[300],
             scales=None, sizes=[300]),
        dict(input_shape=[3], output_shape=[1],
             scales=None, sizes=[1]),
    ]

    @pytest.mark.parametrize("params", test_data_sizes_nearest)
    @pytest.mark.parametrize("coordinate_transformation_mode",
                             ['half_pixel', 'pytorch_half_pixel', 'align_corners',
                              'asymmetric', 'tf_half_pixel_for_nn'])
    @pytest.mark.parametrize("cubic_coeff_a", [-0.75])
    @pytest.mark.parametrize("mode", ['nearest'])
    @pytest.mark.parametrize("nearest_mode", ['round_prefer_floor', 'round_prefer_ceil',
                                              'floor', 'ceil'])
    def test_resize_combined_sizes_nearest(self, params, coordinate_transformation_mode,
                                           cubic_coeff_a, mode,
                                           nearest_mode, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_resize_net(**params,
                                           coordinate_transformation_mode=coordinate_transformation_mode,
                                           cubic_coeff_a=cubic_coeff_a, mode=mode,
                                           nearest_mode=nearest_mode,
                                           precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_sizes_linear = [
        dict(input_shape=[1, 3, 100, 200], output_shape=[1, 3, 350, 150],
             scales=None, sizes=[1, 3, 350, 150]),
        dict(input_shape=[16, 7, 190, 400], output_shape=[16, 7, 390, 600],
             scales=None, sizes=[16, 7, 390, 600]),
        dict(input_shape=[4, 33, 600, 800], output_shape=[4, 33, 300, 800],
             scales=None, sizes=[4, 33, 300, 800]),
        dict(input_shape=[4, 33, 3, 800], output_shape=[4, 33, 1, 800],
             scales=None, sizes=[4, 33, 1, 800]),
        dict(input_shape=[100, 200], output_shape=[350, 150],
             scales=None, sizes=[350, 150]),
        dict(input_shape=[190, 400], output_shape=[390, 600],
             scales=None, sizes=[390, 600]),
        dict(input_shape=[600, 800], output_shape=[300, 800],
             scales=None, sizes=[300, 800]),
        dict(input_shape=[3, 800], output_shape=[1, 800],
             scales=None, sizes=[1, 800]),
    ]

    @pytest.mark.parametrize("params", test_data_sizes_linear)
    @pytest.mark.parametrize("coordinate_transformation_mode",
                             ['half_pixel', 'pytorch_half_pixel', 'align_corners',
                              'asymmetric', 'tf_half_pixel_for_nn'])
    @pytest.mark.parametrize("cubic_coeff_a", [-0.75])
    @pytest.mark.parametrize("mode", ['linear'])
    @pytest.mark.parametrize("nearest_mode", ['round_prefer_floor'])
    def test_resize_combined_sizes_linear(self, params, coordinate_transformation_mode,
                                          cubic_coeff_a, mode,
                                          nearest_mode, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_resize_net(**params,
                                           coordinate_transformation_mode=coordinate_transformation_mode,
                                           cubic_coeff_a=cubic_coeff_a, mode=mode,
                                           nearest_mode=nearest_mode,
                                           precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, custom_eps=2.0e-2, temp_dir=temp_dir)


def create_ref_net_in_sizes_mode(precision, input_shape, output_shape, sizes_value, scales_value,
                                 attrs):
    input_data_type = np_data_type_to_destination_type(data_type_str_to_np(precision))
    input_rank = len(input_shape)
    epsilon = np.array([1.0e-5])
    spatial_dims = spatial_dimensions(input_shape)
    begin_dim = spatial_dims[0]
    end_dim = input_rank

    spatial_sizes_value = sizes_value[spatial_dims]

    nodes_attrs = {
        'input': {'kind': 'op', 'type': 'Parameter'},
        'input_data': {'shape': input_shape, 'kind': 'data'},
        'shape_of': {'kind': 'op', 'type': 'ShapeOf'},
        'shape_of_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'shape_to_float': {'kind': 'op', 'type': 'Convert', 'destination_type': input_data_type},
        'shape_to_float_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'div': {'kind': 'op', 'type': 'Divide'},
        'div_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'div_sizes_const_data': {'kind': 'data', 'value': sizes_value},
        'div_sizes_const': {'kind': 'op', 'type': 'Const'},
        'div_sizes_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'eps_const_data': {'kind': 'data', 'value': epsilon},
        'eps_const': {'kind': 'op', 'type': 'Const'},
        'eps_data': {'shape': int64_array([1]), 'kind': 'data'},
        'add': {'kind': 'op', 'type': 'Add'},
        'add_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'ss_scales': {
            'kind': 'op', 'type': 'StridedSlice', 'begin_mask': 0,
            'end_mask': 0, 'new_axis_mask': 0,
            'shrink_axis_mask': 0, 'ellipsis_mask': 0
        },
        'ss_scales_data': {'shape': int64_array([len(spatial_sizes_value)]), 'kind': 'data'},
        'ss_scales_begin_const_data': {'kind': 'data', 'value': int64_array([begin_dim])},
        'ss_scales_begin_const': {'kind': 'op', 'type': 'Const'},
        'ss_scales_begin_data': {'shape': int64_array([1]), 'kind': 'data'},
        'ss_scales_end_const_data': {'kind': 'data', 'value': int64_array([end_dim])},
        'ss_scales_end_const': {'kind': 'op', 'type': 'Const'},
        'ss_scales_end_data': {'shape': int64_array([1]), 'kind': 'data'},
        'ss_scales_stride_const_data': {'kind': 'data', 'value': int64_array([1])},
        'ss_scales_stride_const': {'kind': 'op', 'type': 'Const'},
        'ss_scales_stride_data': {'shape': int64_array([1]), 'kind': 'data'},
        'sizes_const_data': {'kind': 'data', 'value': spatial_sizes_value},
        'sizes_const': {'kind': 'op', 'type': 'Const'},
        'sizes_data': {'shape': int64_array([len(spatial_sizes_value)]), 'kind': 'data'},
        'axes_const_data': {'kind': 'data', 'value': spatial_dims},
        'axes_const': {'kind': 'op', 'type': 'Const'},
        'axes_data': {'shape': int64_array([len(spatial_dims)]), 'kind': 'data'},
        'interpolate': attrs,
        'interpolate_data': {'shape': output_shape, 'kind': 'data'},
        'result': {'kind': 'op', 'type': 'Result'},
    }
    edges = [
        ('input', 'input_data'),
        ('input_data', 'interpolate', {'in': 0, 'out': 0}),
        ('input_data', 'shape_of', {'in': 0, 'out': 0}),
        ('shape_of', 'shape_of_data'),
        ('shape_of_data', 'shape_to_float'),
        ('shape_to_float', 'shape_to_float_data'),
        ('shape_to_float_data', 'div', {'in': 1}),
        ('div_sizes_const_data', 'div_sizes_const'),
        ('div_sizes_const', 'div_sizes_data'),
        ('div_sizes_data', 'div', {'in': 0}),
        ('div', 'div_data'),
        ('eps_const_data', 'eps_const'),
        ('eps_const', 'eps_data'),
        ('div_data', 'add', {'in': 0}),
        ('eps_data', 'add', {'in': 1}),
        ('add', 'add_data'),
        ('add_data', 'ss_scales', {'in': 0}),
        ('ss_scales', 'ss_scales_data'),
        ('ss_scales_begin_const_data', 'ss_scales_begin_const'),
        ('ss_scales_begin_const', 'ss_scales_begin_data'),
        ('ss_scales_begin_data', 'ss_scales', {'in': 1}),
        ('ss_scales_end_const_data', 'ss_scales_end_const'),
        ('ss_scales_end_const', 'ss_scales_end_data'),
        ('ss_scales_end_data', 'ss_scales', {'in': 2}),
        ('ss_scales_stride_const_data', 'ss_scales_stride_const'),
        ('ss_scales_stride_const', 'ss_scales_stride_data'),
        ('ss_scales_stride_data', 'ss_scales', {'in': 3}),
        ('ss_scales_data', 'interpolate', {'in': 2}),
        ('sizes_const_data', 'sizes_const'),
        ('sizes_const', 'sizes_data'),
        ('sizes_data', 'interpolate', {'in': 1}),
        ('axes_const_data', 'axes_const'),
        ('axes_const', 'axes_data'),
        ('axes_data', 'interpolate', {'in': 3}),
        ('interpolate', 'interpolate_data'),
        ('interpolate_data', 'result')
    ]

    return build_graph(nodes_attrs, edges)


def create_ref_net_in_scales_mode(precision, input_shape, output_shape, sizes_value, scales_value,
                                  attrs):
    input_data_type = np_data_type_to_destination_type(data_type_str_to_np(precision))
    input_rank = len(input_shape)
    epsilon = np.array([1.0e-5])
    spatial_dims = spatial_dimensions(input_shape)
    begin_dim = spatial_dims[0]
    end_dim = input_rank

    spatial_scales_value = scales_value[spatial_dims]

    nodes_attrs = {
        'input': {'kind': 'op', 'type': 'Parameter'},
        'input_data': {'shape': input_shape, 'kind': 'data'},
        'shape_of': {'kind': 'op', 'type': 'ShapeOf'},
        'shape_of_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'shape_to_float': {'kind': 'op', 'type': 'Convert', 'destination_type': input_data_type},
        'shape_to_float_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'mul': {'kind': 'op', 'type': 'Multiply'},
        'mul_scales_const_data': {'kind': 'data', 'value': scales_value},
        'mul_scales_const': {'kind': 'op', 'type': 'Const'},
        'mul_scales_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'mul_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'eps_const_data': {'kind': 'data', 'value': epsilon},
        'eps_const': {'kind': 'op', 'type': 'Const'},
        'eps_data': {'shape': int64_array([1]), 'kind': 'data'},
        'add': {'kind': 'op', 'type': 'Add'},
        'add_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'floor': {'type': 'Floor', 'kind': 'op'},
        'floor_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'to_int': {'kind': 'op', 'type': 'Convert', 'destination_type': 'i64'},
        'to_int_data': {'shape': int64_array([input_rank]), 'kind': 'data'},
        'strided_slice': {
            'kind': 'op', 'type': 'StridedSlice', 'begin_mask': 0,
            'end_mask': 0, 'new_axis_mask': 0,
            'shrink_axis_mask': 0, 'ellipsis_mask': 0
        },
        'strided_slice_data': {'shape': int64_array([len(spatial_scales_value)]), 'kind': 'data'},
        'begin_const_data': {'kind': 'data', 'value': int64_array([begin_dim])},
        'begin_const': {'kind': 'op', 'type': 'Const'},
        'begin_data': {'shape': int64_array([1]), 'kind': 'data'},
        'end_const_data': {'kind': 'data', 'value': int64_array([end_dim])},
        'end_const': {'kind': 'op', 'type': 'Const'},
        'end_data': {'shape': int64_array([1]), 'kind': 'data'},
        'stride_const_data': {'kind': 'data', 'value': int64_array([1])},
        'stride_const': {'kind': 'op', 'type': 'Const'},
        'stride_data': {'shape': int64_array([1]), 'kind': 'data'},
        'scales_const_data': {'kind': 'data', 'value': spatial_scales_value},
        'scales_const': {'kind': 'op', 'type': 'Const'},
        'scales_data': {'shape': int64_array([len(spatial_scales_value)]), 'kind': 'data'},
        'axes_const_data': {'kind': 'data', 'value': spatial_dims},
        'axes_const': {'kind': 'op', 'type': 'Const'},
        'axes_data': {'shape': int64_array([len(spatial_dims)]), 'kind': 'data'},
        'interpolate': attrs,
        'interpolate_data': {'shape': output_shape, 'kind': 'data'},
        'result': {'kind': 'op', 'type': 'Result'},
    }
    edges = [
        ('input', 'input_data'),
        ('input_data', 'interpolate', {'in': 0, 'out': 0}),
        ('input_data', 'shape_of', {'in': 0, 'out': 0}),
        ('shape_of', 'shape_of_data'),
        ('shape_of_data', 'shape_to_float'),
        ('shape_to_float', 'shape_to_float_data'),
        ('shape_to_float_data', 'mul', {'in': 0}),
        ('mul_scales_const_data', 'mul_scales_const'),
        ('mul_scales_const', 'mul_scales_data'),
        ('mul_scales_data', 'mul', {'in': 1}),
        ('mul', 'mul_data'),
        ('eps_const_data', 'eps_const'),
        ('eps_const', 'eps_data'),
        ('mul_data', 'add', {'in': 0}),
        ('eps_data', 'add', {'in': 1}),
        ('add', 'add_data'),
        ('add_data', 'floor'),
        ('floor', 'floor_data'),
        ('floor_data', 'to_int'),
        ('to_int', 'to_int_data'),
        ('to_int_data', 'strided_slice', {'in': 0}),
        ('strided_slice', 'strided_slice_data'),
        ('begin_const_data', 'begin_const'),
        ('begin_const', 'begin_data'),
        ('begin_data', 'strided_slice', {'in': 1}),
        ('end_const_data', 'end_const'),
        ('end_const', 'end_data'),
        ('end_data', 'strided_slice', {'in': 2}),
        ('stride_const_data', 'stride_const'),
        ('stride_const', 'stride_data'),
        ('stride_data', 'strided_slice', {'in': 3}),
        ('strided_slice_data', 'interpolate', {'in': 1}),
        ('scales_const_data', 'scales_const'),
        ('scales_const', 'scales_data'),
        ('scales_data', 'interpolate', {'in': 2}),
        ('axes_const_data', 'axes_const'),
        ('axes_const', 'axes_data'),
        ('axes_data', 'interpolate', {'in': 3}),
        ('interpolate', 'interpolate_data'),
        ('interpolate_data', 'result')
    ]

    return build_graph(nodes_attrs, edges)


def spatial_dimensions(shape):
    rank = len(shape)
    if rank >= 4:
        return np.arange(2, rank)
    elif rank in [1, 2]:
        return np.arange(0, rank)
    else:
        return np.arange(1, rank)


def convert_onnx_mode(mode: str) -> str:
    return {'nearest': 'nearest', 'linear': 'linear_onnx', 'cubic': 'cubic'}[mode]

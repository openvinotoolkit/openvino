# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace
from unittest.mock import patch
import os

import pytest

from openvino.tools.mo.front.tf.ObjectDetectionAPI import calculate_shape_keeping_aspect_ratio, \
    calculate_placeholder_spatial_shape, ObjectDetectionAPIPreprocessor2Replacement
from openvino.tools.mo.front.common.partial_infer.utils import float32_array
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.custom_replacement_config import CustomReplacementDescriptor
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.mo.utils.pipeline_config_test import file_content
from unit_tests.utils.graph import const, regular_op, result, build_graph, connect_front
from openvino.runtime import PartialShape


class FakePipelineConfig:
    def __init__(self, model_params: dict):
        self._model_params = model_params

    def get_param(self, param: str):
        if param not in self._model_params:
            return None
        return self._model_params[param]


class TestCalculateShape():
    min_size = 600
    max_size = 1024

    @pytest.mark.parametrize("h, w, th, tw",[(100, 300, 341, 1024),
                (100, 600, 171, 1024),
                (100, 3000, 34, 1024),
                (300, 300, 600, 600),
                (300, 400, 600, 800),
                (300, 600, 512, 1024),
                (1000, 2500, 410, 1024),
                (1800, 2000, 600, 667),
                (300, 100, 1024, 341),
                (600, 100, 1024, 171),
                (3000, 100, 1024, 34),
                (400, 300, 800, 600),
                (600, 300, 1024, 512),
                (2500, 1000, 1024, 410),
                (2000, 1800, 667, 600),
                ])
    def test_calculate_shape(self, h, w, th, tw):
        assert calculate_shape_keeping_aspect_ratio(h, w, self.min_size, self.max_size) == (th, tw)


class TestCalculatePlaceholderSpatialShape(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.graph.graph['user_shapes'] = None
        self.replacement_desc = CustomReplacementDescriptor('dummy_id', {})
        self.match = SubgraphMatch(self.graph, self.replacement_desc, [], [], [], '')
        self.pipeline_config = FakePipelineConfig({})

    def test_default_fixed_shape_resizer(self):
        self.pipeline_config._model_params['resizer_image_height'] = 300
        self.pipeline_config._model_params['resizer_image_width'] = 600
        self.assertTupleEqual((300, 600),
                              calculate_placeholder_spatial_shape(self.graph, self.match, self.pipeline_config))

    def test_fixed_shape_resizer_overrided_by_user(self):
        self.pipeline_config._model_params['resizer_image_height'] = 300
        self.pipeline_config._model_params['resizer_image_width'] = 600
        self.graph.graph['user_shapes'] = {'image_tensor': [{'shape': PartialShape([1, 400, 500, 3])}]}
        self.assertTupleEqual((400, 500),
                              calculate_placeholder_spatial_shape(self.graph, self.match, self.pipeline_config))

    def test_default_keep_aspect_ratio_resizer(self):
        self.pipeline_config._model_params['resizer_min_dimension'] = 600
        self.pipeline_config._model_params['resizer_max_dimension'] = 1024
        self.assertTupleEqual((600, 600),
                              calculate_placeholder_spatial_shape(self.graph, self.match, self.pipeline_config))

    def test_keep_aspect_ratio_resizer_overrided_by_user(self):
        self.pipeline_config._model_params['resizer_min_dimension'] = 600
        self.pipeline_config._model_params['resizer_max_dimension'] = 1024
        self.graph.graph['user_shapes'] = {'image_tensor': [{'shape': PartialShape([1, 400, 300, 3])}]}
        self.assertTupleEqual((800, 600),
                              calculate_placeholder_spatial_shape(self.graph, self.match, self.pipeline_config))

    def test_keep_aspect_ratio_resizer_overrided_by_user_pad(self):
        self.pipeline_config._model_params['resizer_min_dimension'] = 600
        self.pipeline_config._model_params['resizer_max_dimension'] = 1024
        self.pipeline_config._model_params['pad_to_max_dimension'] = True
        self.graph.graph['user_shapes'] = {'image_tensor': [{'shape': PartialShape([1, 400, 300, 3])}]}
        self.assertTupleEqual((1024, 1024),
                              calculate_placeholder_spatial_shape(self.graph, self.match, self.pipeline_config))

    def test_missing_input_shape_information(self):
        self.assertRaises(Error, calculate_placeholder_spatial_shape, self.graph, self.match, self.pipeline_config)


@patch('openvino.tools.mo.front.tf.ObjectDetectionAPI.update_parameter_shape')
class TestObjectDetectionAPIPreprocessor2Replacement(unittest.TestCase):
    def setUp(self):
        self.start_node_name = 'StatefulPartitionedCall/Preprocessor/unstack'
        self.end_node_name = 'StatefulPartitionedCall/Preprocessor/stack'
        self.end_node_name2 = 'StatefulPartitionedCall/Preprocessor/stack2'
        self.loop_start_node_name = 'prefix/map/while/Preprocessor/unstack'
        self.loop_end_node_name = 'prefix/map/while/Preprocessor/stack'
        self.mul_const = float32_array([0.025, 0.374, -0.45])
        self.sub_const = float32_array([2.0, 3.0, 4.0])

        self.nodes = {
            **regular_op('input', {'op': 'Parameter', 'type': 'Parameter'}),

            **regular_op('mul', {'op': 'Mul', 'type': 'Multiply', 'name': 'my_mul'}),
            **regular_op('sub', {'op': 'Sub', 'type': 'Subtract', 'name': 'my_sub'}),
            **const('mul_const', self.mul_const),
            **const('sub_const', self.sub_const),

            **regular_op(self.start_node_name, {'op': 'Identity'}),
            **regular_op(self.end_node_name, {'op': 'Identity'}),
            **regular_op(self.end_node_name2, {'op': 'Identity'}),

            **regular_op('loop', {'op': 'Loop', 'body': None}),

            **regular_op('resize', {'type': 'Interpolate'}),
            **result('result'),
        }
        self.replacement_desc = {'start_nodes': [self.start_node_name],
                                 'end_nodes': [self.end_node_name, self.end_node_name2]}

    def build_ref_graph(self, preprocessing: bool):
        if preprocessing:
            ref_edges = [*connect_front('input', '0:mul'),
                         *connect_front('mul_const', '1:mul'),
                         *connect_front('sub_const', '0:sub'),
                         *connect_front('mul', '1:sub'),
                         *connect_front('sub', 'result'),
                         ]
        else:
            ref_edges = [*connect_front('input', 'result')]
        ref_graph = build_graph(self.nodes, ref_edges, nodes_with_edges_only=True)
        ref_graph.stage = 'front'
        return ref_graph

    def test_case_1_pad_to_max_dim(self, update_parameter_shape_mock):
        # test for case #1 described in the ObjectDetectionAPIPreprocessor2Replacement
        # sub/mul should be removed because they are applied before prep-processing and pad_to_max_dimension is True
        update_parameter_shape_mock.return_value = (None, None)
        edges = [*connect_front('input', '0:mul'),
                 *connect_front('mul_const', '1:mul'),
                 *connect_front('sub_const', '0:sub'),
                 *connect_front('mul', '1:sub'),
                 *connect_front('sub', self.start_node_name),
                 *connect_front(self.start_node_name, 'resize'),
                 *connect_front('resize', self.end_node_name),
                 *connect_front(self.end_node_name, 'result'),
                 ]
        graph = build_graph(self.nodes, edges)
        graph.stage = 'front'
        graph.graph['cmd_params'] = Namespace(tensorflow_object_detection_api_pipeline_config=__file__)

        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=file_content)):
            ObjectDetectionAPIPreprocessor2Replacement().transform_graph(graph, self.replacement_desc)

        (flag, resp) = compare_graphs(graph, self.build_ref_graph(False), 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_case_1_no_pad_to_max_dim(self, update_parameter_shape_mock):
        # test for case #1 described in the ObjectDetectionAPIPreprocessor2Replacement
        # sub/mul should be kept even though they are applied before prep-processing and pad_to_max_dimension is False
        update_parameter_shape_mock.return_value = (None, None)
        edges = [*connect_front('input', '0:mul'),
                 *connect_front('mul_const', '1:mul'),
                 *connect_front('sub_const', '0:sub'),
                 *connect_front('mul', '1:sub'),
                 *connect_front('sub', self.start_node_name),
                 *connect_front(self.start_node_name, 'resize'),
                 *connect_front('resize', self.end_node_name),
                 *connect_front(self.end_node_name, 'result'),
                 ]
        graph = build_graph(self.nodes, edges)
        graph.stage = 'front'
        graph.graph['cmd_params'] = Namespace(tensorflow_object_detection_api_pipeline_config=__file__)

        updated_pipeline_config_content = file_content.replace('pad_to_max_dimension: true',
                                                               'pad_to_max_dimension: false')
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=updated_pipeline_config_content)):
            ObjectDetectionAPIPreprocessor2Replacement().transform_graph(graph, self.replacement_desc)

        (flag, resp) = compare_graphs(graph, self.build_ref_graph(True), 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_case_2(self, update_parameter_shape_mock):
        # test for case #2 described in the ObjectDetectionAPIPreprocessor2Replacement
        update_parameter_shape_mock.return_value = (None, None)

        edges = [*connect_front('input', self.start_node_name),
                 *connect_front(self.start_node_name, 'resize'),
                 *connect_front('resize', self.end_node_name),
                 *connect_front(self.end_node_name, '0:mul'),
                 *connect_front('mul_const', '1:mul'),
                 *connect_front('sub_const', '0:sub'),
                 *connect_front('mul', '1:sub'),
                 *connect_front('sub', 'result'),
                 ]
        graph = build_graph(self.nodes, edges)
        graph.stage = 'front'
        graph.graph['cmd_params'] = Namespace(tensorflow_object_detection_api_pipeline_config=__file__)

        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=file_content)):
            ObjectDetectionAPIPreprocessor2Replacement().transform_graph(graph, self.replacement_desc)

        (flag, resp) = compare_graphs(graph, self.build_ref_graph(True), 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_case_3(self, update_parameter_shape_mock):
        # test for case #3 described in the ObjectDetectionAPIPreprocessor2Replacement
        update_parameter_shape_mock.return_value = (None, None)

        edges = [*connect_front('input', self.start_node_name),
                 *connect_front(self.start_node_name, 'resize'),
                 *connect_front('resize', self.end_node_name),
                 *connect_front(self.end_node_name, 'result'),
                 ]
        graph = build_graph(self.nodes, edges)
        graph.stage = 'front'
        graph.graph['cmd_params'] = Namespace(tensorflow_object_detection_api_pipeline_config=__file__)

        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=file_content)):
            ObjectDetectionAPIPreprocessor2Replacement().transform_graph(graph, self.replacement_desc)

        (flag, resp) = compare_graphs(graph, self.build_ref_graph(False), 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def build_main_graph(self, pre_processing: str):
        def build_body_graph(pre_processing: str):
            nodes = {
                **regular_op('input', {'type': 'Parameter', 'op': 'Parameter'}),

                **regular_op('mul', {'op': 'Mul', 'type': 'Multiply', 'name': 'my_body_mul'}),
                **regular_op('sub', {'op': 'Sub', 'type': 'Subtract', 'name': 'my_body_sub'}),
                **const('body_mul_const', self.mul_const),
                **const('body_sub_const', self.sub_const),

                **regular_op(self.loop_start_node_name, {'op': 'Identity'}),
                **regular_op(self.loop_end_node_name, {'op': 'Identity'}),

                **regular_op('resize', {'type': 'Interpolate'}),
                **result('result'),
            }
            if pre_processing == 'no':
                edges = [*connect_front('input', self.loop_start_node_name),
                         *connect_front(self.loop_start_node_name, 'resize'),
                         *connect_front('resize', self.loop_end_node_name),
                         *connect_front(self.loop_end_node_name, 'result'),
                         ]
            elif pre_processing == 'trailing':
                edges = [*connect_front('input', self.loop_start_node_name),
                         *connect_front(self.loop_start_node_name, 'resize'),
                         *connect_front('resize', self.loop_end_node_name),
                         *connect_front(self.loop_end_node_name, '0:mul'),
                         *connect_front('body_mul_const', '1:mul'),
                         *connect_front('body_sub_const', '0:sub'),
                         *connect_front('mul', '1:sub'),
                         *connect_front('sub', 'result'),
                         ]
            else:
                edges = [*connect_front('input', '0:mul'),
                         *connect_front('body_mul_const', '1:mul'),
                         *connect_front('body_sub_const', '0:sub'),
                         *connect_front('mul', '1:sub'),
                         *connect_front('sub', self.loop_start_node_name),
                         *connect_front(self.loop_start_node_name, 'resize'),
                         *connect_front('resize', self.loop_end_node_name),
                         *connect_front(self.loop_end_node_name, 'result'),
                         ]
            graph = build_graph(nodes, edges, nodes_with_edges_only=True)
            graph.stage = 'front'
            return graph

        edges = [*connect_front('input', self.start_node_name),
                 *connect_front(self.start_node_name, 'loop'),
                 *connect_front('loop:0', self.end_node_name),
                 *connect_front('loop:1', self.end_node_name2),
                 *connect_front(self.end_node_name, 'result'),
                 ]
        graph = build_graph(self.nodes, edges, {'loop': {'body': build_body_graph(pre_processing)}},
                            nodes_with_edges_only=True)
        graph.stage = 'front'
        return graph

    def test_case_4(self, update_parameter_shape_mock):
        # test for case #4 described in the ObjectDetectionAPIPreprocessor2Replacement
        update_parameter_shape_mock.return_value = (None, None)

        graph = self.build_main_graph('leading')
        graph.graph['cmd_params'] = Namespace(tensorflow_object_detection_api_pipeline_config=__file__)

        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=file_content)):
            ObjectDetectionAPIPreprocessor2Replacement().transform_graph(graph, self.replacement_desc)

        (flag, resp) = compare_graphs(graph, self.build_ref_graph(True), 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_case_5(self, update_parameter_shape_mock):
        # test for case #5 described in the ObjectDetectionAPIPreprocessor2Replacement
        update_parameter_shape_mock.return_value = (None, None)

        graph = self.build_main_graph('trailing')
        graph.graph['cmd_params'] = Namespace(tensorflow_object_detection_api_pipeline_config=__file__)

        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=file_content)):
            ObjectDetectionAPIPreprocessor2Replacement().transform_graph(graph, self.replacement_desc)

        (flag, resp) = compare_graphs(graph, self.build_ref_graph(True), 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_case_6(self, update_parameter_shape_mock):
        # test for case #6 described in the ObjectDetectionAPIPreprocessor2Replacement
        update_parameter_shape_mock.return_value = (None, None)

        graph = self.build_main_graph('no')
        graph.graph['cmd_params'] = Namespace(tensorflow_object_detection_api_pipeline_config=__file__)

        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=file_content)):
            ObjectDetectionAPIPreprocessor2Replacement().transform_graph(graph, self.replacement_desc)

        (flag, resp) = compare_graphs(graph, self.build_ref_graph(False), 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)


class TestPipelineConfig(unittest.TestCase):
    def test_pipeline_config_loading(self):
        from openvino.tools.mo.utils.pipeline_config import PipelineConfig
        pipeline_config = PipelineConfig(os.path.join(os.path.dirname(__file__), "test_configs/config1.config"))
        assert pipeline_config.get_param('ssd_anchor_generator_num_layers') == 6
        assert pipeline_config.get_param('num_classes') == 90
        assert pipeline_config.get_param('resizer_image_width') == 300
        assert pipeline_config.get_param('resizer_image_height') == 300

        pipeline_config = PipelineConfig(os.path.join(os.path.dirname(__file__), "test_configs/config2.config"))
        assert pipeline_config.get_param('ssd_anchor_generator_num_layers') is None
        assert pipeline_config.get_param('num_classes') == 10
        assert pipeline_config.get_param('resizer_image_width') == 640
        assert pipeline_config.get_param('resizer_image_height') == 640

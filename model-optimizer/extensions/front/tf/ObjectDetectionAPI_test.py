"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.front.tf.ObjectDetectionAPI import calculate_shape_keeping_aspect_ratio, \
    calculate_placeholder_spatial_shape
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph
from mo.utils.custom_replacement_config import CustomReplacementDescriptor
from mo.utils.error import Error


class FakePipelineConfig:
    def __init__(self, model_params: dict):
        self._model_params = model_params

    def get_param(self, param: str):
        if param not in self._model_params:
            return None
        return self._model_params[param]


class TestCalculateShape(unittest.TestCase):
    min_size = 600
    max_size = 1024

    def test_calculate_shape_1(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(100, 300, self.min_size, self.max_size), (341, 1024))

    def test_calculate_shape_2(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(100, 600, self.min_size, self.max_size), (171, 1024))

    def test_calculate_shape_3(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(100, 3000, self.min_size, self.max_size), (34, 1024))

    def test_calculate_shape_4(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(300, 300, self.min_size, self.max_size), (600, 600))

    def test_calculate_shape_5(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(300, 400, self.min_size, self.max_size), (600, 800))

    def test_calculate_shape_6(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(300, 600, self.min_size, self.max_size), (512, 1024))

    def test_calculate_shape_7(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(1000, 2500, self.min_size, self.max_size),
                              (410, 1024))

    def test_calculate_shape_8(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(1800, 2000, self.min_size, self.max_size),
                              (600, 667))

    def test_calculate_shape_11(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(300, 100, self.min_size, self.max_size), (1024, 341))

    def test_calculate_shape_12(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(600, 100, self.min_size, self.max_size), (1024, 171))

    def test_calculate_shape_13(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(3000, 100, self.min_size, self.max_size), (1024, 34))

    def test_calculate_shape_15(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(400, 300, self.min_size, self.max_size), (800, 600))

    def test_calculate_shape_16(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(600, 300, self.min_size, self.max_size), (1024, 512))

    def test_calculate_shape_17(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(2500, 1000, self.min_size, self.max_size),
                              (1024, 410))

    def test_calculate_shape_18(self):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(2000, 1800, self.min_size, self.max_size),
                              (667, 600))


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
        self.graph.graph['user_shapes'] = {'image_tensor': [{'shape': [1, 400, 500, 3]}]}
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
        self.graph.graph['user_shapes'] = {'image_tensor': [{'shape': [1, 400, 300, 3]}]}
        self.assertTupleEqual((800, 600),
                              calculate_placeholder_spatial_shape(self.graph, self.match, self.pipeline_config))

    def test_missing_input_shape_information(self):
        self.assertRaises(Error, calculate_placeholder_spatial_shape, self.graph, self.match, self.pipeline_config)

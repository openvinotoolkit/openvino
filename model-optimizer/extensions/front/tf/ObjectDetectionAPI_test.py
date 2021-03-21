"""
 Copyright (C) 2018-2021 Intel Corporation

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

from generator import generator, generate

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


@generator
class TestCalculateShape(unittest.TestCase):
    min_size = 600
    max_size = 1024

    @generate(*[(100, 300, 341, 1024, False),
                (100, 600, 171, 1024, False),
                (100, 3000, 34, 1024, False),
                (300, 300, 600, 600, False),
                (300, 400, 600, 800, False),
                (300, 600, 512, 1024, False),
                (1000, 2500, 410, 1024, False),
                (1800, 2000, 600, 667, False),
                (300, 100, 1024, 341, False),
                (600, 100, 1024, 171, False),
                (3000, 100, 1024, 34, False),
                (400, 300, 800, 600, False),
                (600, 300, 1024, 512, False),
                (2500, 1000, 1024, 410, False),
                (2000, 1800, 667, 600, False),
                (300, 300, 1024, 1024, True),
                (900, 300, 1024, 1024, True),
                (1300, 900, 1024, 1024, True),
                (1025, 1025, 1024, 1024, True),
                ])
    def test_calculate_shape(self, h, w, th, tw, pad):
        self.assertTupleEqual(calculate_shape_keeping_aspect_ratio(h, w, self.min_size, self.max_size, pad), (th, tw))


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

    def test_keep_aspect_ratio_resizer_overrided_by_user_pad(self):
        self.pipeline_config._model_params['resizer_min_dimension'] = 600
        self.pipeline_config._model_params['resizer_max_dimension'] = 1024
        self.pipeline_config._model_params['pad_to_max_dimension'] = True
        self.graph.graph['user_shapes'] = {'image_tensor': [{'shape': [1, 400, 300, 3]}]}
        self.assertTupleEqual((1024, 1024),
                              calculate_placeholder_spatial_shape(self.graph, self.match, self.pipeline_config))

    def test_missing_input_shape_information(self):
        self.assertRaises(Error, calculate_placeholder_spatial_shape, self.graph, self.match, self.pipeline_config)

"""
 Copyright (c) 2019 Intel Corporation

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

import logging as log

import numpy as np

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class RemoveRedundantReshapeAfterCropAndResize(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def run_before(self):
        return []

    def pattern(self):
        return dict(
            nodes=[
                ('crop_and_resize', dict(kind='op', op='CropAndResize')),
                ('crop_and_resize_data', dict(kind='data')),
                ('reshape_1', dict(kind='op', op='Reshape')),
                ('reshape_1_data', dict(kind='data')),
                ('reshape_2', dict(kind='op', op='Reshape')),
            ],
            edges=[
                ('crop_and_resize', 'crop_and_resize_data'),
                ('crop_and_resize_data', 'reshape_1'),
                ('reshape_1', 'reshape_1_data'),
                ('reshape_1_data', 'reshape_2'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        car_node = match['crop_and_resize']
        reshape_2_node = match['reshape_2']

        shape_1 = match['crop_and_resize_data'].shape
        shape_2 = match['reshape_2'].out_node().shape
        if not np.all(shape_1 == shape_2):
            log.debug('Cannot remove reshape operations after CropAndResize due to different shapes: {} vs {}'.format(
                shape_1, shape_2
            ))
            return

        car_node.out_port(0).disconnect()
        consumer_port_node = reshape_2_node.out_port(0).get_connection().get_destination()
        consumer_port_node.disconnect()
        car_node.out_port(0).connect(consumer_port_node)

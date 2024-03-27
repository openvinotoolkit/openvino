# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.middle.FuseReshapesSequence import FuseReshapesSequence
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class RemoveRedundantReshapeAfterCropAndResize(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def run_before(self):
        return [FuseReshapesSequence]

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

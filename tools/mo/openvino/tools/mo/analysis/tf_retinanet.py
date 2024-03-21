# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.pattern_match import apply_pattern
from openvino.tools.mo.utils.model_analysis import AnalyzeAction


def pattern_instance_counter(graph: Graph, match: dict):
    pattern_instance_counter.counter += 1


pattern_instance_counter.counter = 0

RETINANET_PATTERN = {
    'nodes': [
        ('range_1', dict(kind='op', op='Range')),
        ('range_2', dict(kind='op', op='Range')),
        ('cast_1', dict(kind='op', op='Cast')),
        ('cast_2', dict(kind='op', op='Cast')),
        ('add_1', dict(kind='op', op='Add')),
        ('add_2', dict(kind='op', op='Add')),
        ('mul_1', dict(kind='op', op='Mul')),
        ('mul_2', dict(kind='op', op='Mul')),
        ('size_1', dict(kind='op', op='Size')),
        ('size_2', dict(kind='op', op='Size')),
        ('pack', dict(kind='op', op='Pack')),
        ('fill', dict(kind='op', op='Fill'))
    ],

    'edges': [
        ('range_1', 'cast_1'),
        ('range_2', 'cast_2'),
        ('cast_1', 'add_1'),
        ('cast_2', 'add_2'),
        ('add_1', 'mul_1'),
        ('add_2', 'mul_2'),
        ('mul_1', 'size_1'),
        ('mul_2', 'size_2'),
        ('size_1', 'pack'),
        ('size_2', 'pack'),
        ('pack', 'fill')
    ]
}


class TensorFlowRetinaNet(AnalyzeAction):

    def analyze(self, graph: Graph):
        pattern_instance_counter.counter = 0
        apply_pattern(graph, **RETINANET_PATTERN, action=pattern_instance_counter)

        if pattern_instance_counter.counter > 0:
            result = dict()
            result['mandatory_parameters'] = {'transformations_config':
                                                  'openvino/tools/mo/front/tf/retinanet.json'}

            message = "Your model looks like TensorFlow RetinaNet Model.\n" \
                      "To generate the IR, provide model to the Model Optimizer with the following parameters:\n" \
                      "\t--input_model <path_to_model>/<model>.pb\n" \
                      "\t--input_shape [1,600,600,3]\n" \
                      "\t--transformations_config <OPENVINO_INSTALL_DIR>/tools/model_optimizer/openvino/tools/mo/front/tf/retinanet.json\n" \
                      "\t--reverse_input_channels"

            return {'model_type': {'TF_RetinaNet': result}}, message

        return None, None

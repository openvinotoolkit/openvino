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

from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.result import Result


class TopKNormalizer(BackReplacementPattern):
    """
    The transformation converts the second input to the TopK layer from 0D to 1D.
    TODO this pass should be removed when IE suport 0D tensors.
    """
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('result', {'type': 'TopK'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['result']
        k = node.in_port(1).data.get_value()
        if not isinstance(k, np.ndarray):
            node.in_port(1).data.set_value(int64_array([k]))
        elif k.ndim == 0:
            node.in_port(1).data.set_value(int64_array([k.item()]))
        else:
            log.debug('The "k" input to the TopK layer "{}" is already 1D'.format(node.soft_get('name')))

        if node.out_port(0).disconnected():
            output = Result(graph, {'name': node.name + '/Result_port_0/'}).create_node()
            node.out_port(0).get_connection().set_destination(output.in_port(0))
        if node.out_port(1).disconnected():
            output = Result(graph, {'name': node.name + '/Result_port_1/'}).create_node()
            node.out_port(1).get_connection().set_destination(output.in_port(0))

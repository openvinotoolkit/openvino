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

from mo.front.common.replacement import FrontReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.unsqueeze import Unsqueeze


class ExpandDimsToUnsqueeze(FrontReplacementPattern):
    """
    Converts the 'ExpandDims' layer to Unsqueeze layer with two inputs: the input with data and input with the
    dimensions to unsqueeze.
    """
    enabled = True

    def run_after(self):
        from extensions.front.Pack import Pack
        return [Pack]

    def find_and_replace_pattern(self, graph: Graph):
        for expand_dims_node in graph.get_op_nodes(op='ExpandDims'):
            if len(expand_dims_node.in_nodes()) == 1:
                expand_axis = expand_dims_node.expand_axis
                if not isinstance(expand_axis, np.ndarray):
                    expand_axis = int64_array([expand_axis]).flatten()
                unsqueeze_node = Unsqueeze(graph, {'name': expand_dims_node.id}).create_node()
                unsqueeze_dims_node = Const(graph, {'name': expand_dims_node.id + '/Dims',
                                                    'value': expand_axis}).create_node()
                expand_dims_node.in_port(0).get_connection().set_destination(unsqueeze_node.in_port(0))
                expand_dims_node.out_port(0).get_connection().set_source(unsqueeze_node.out_port(0))
                unsqueeze_node.in_port(1).connect(unsqueeze_dims_node.out_port(0))
            else:
                log.error('The ExpandDims node {} has more than 1 input'.format(expand_dims_node.soft_get('name')))

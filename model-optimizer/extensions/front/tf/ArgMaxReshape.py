"""
 Copyright (c) 2018 Intel Corporation

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
import networkx as nx
import numpy as np

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node
from mo.ops.squeeze import Squeeze
from mo.graph.graph import insert_node_after


class ArgMaxReshape(FrontReplacementOp):
    """
    The TensorFlow version of ArgMax removes the reduction axis, but Inference Engine ones doesn't. So this pass adds
    Squeeze to remove the reduction dimension.
    """
    op = "ArgMax"
    enabled = True

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: dict):
        # do not remove matched node
        return []

    def replace_op(self, graph: nx.MultiDiGraph, node: Node):
        squeeze_op = Squeeze(graph, dict())
        squeeze_op.attrs['old_infer'] = squeeze_op.attrs['infer']
        squeeze_op.attrs['infer'] = __class__.do_infer

        squeeze_node = squeeze_op.create_node([], dict(name=node.name + '/Squeeze'))
        insert_node_after(node, squeeze_node)
        return []

    @staticmethod
    def do_infer(node: Node):
        """
        The infer function for Squeeze that get's axis for reduction from the ArgMax node
        """
        argmax_node = node.in_node(0).in_node(0)
        assert argmax_node.type == 'ArgMax'
        if not argmax_node.has('axis'):
            log.error('The node "{}" does not have attribute "axis"'.format(argmax_node.name))
            return
        node['squeeze_dims'] = np.array([argmax_node.axis])
        node.old_infer(node)

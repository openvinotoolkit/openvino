"""
 Copyright (C) 2017-2020 Intel Corporation

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
from copy import copy, deepcopy

from extensions.ops.parameter import Parameter
from mo.graph.graph import Node, dict_includes, Graph
from mo.graph.port import Port
from mo.ops.const import Const
from mo.ops.op import Op
from mo.utils.error import Error
from extensions.ops.tensor_iterator import TensorIterator


class Loop(TensorIterator):
    """
    Loop layer that iterates over tensors and execute embedded sub-graph. The main difference from the TensorIterator is
    that Loop operation performs implicit slicing of data using special input called "current_iteration". Also the Loop
    has special input determining the execution condition and special output producing execution condition for the next
    iteration.
    """

    op = 'Loop'

    def __init__(self, graph: Graph, attrs: dict):
        base_attrs = {
            'type': self.op,
            'op': self.op,
            'version': 'opset5',
            'input_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'output_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'back_edges': [],  # a list of dicts with such attrs as from_layer, from_port, etc.
            'body': None,  # an Graph object with a body sub-graph
            'sub_graphs': [],  # built-in attribute with all sub-graphs
            'infer': self.infer,
            'type_infer': self.ti_type_infer,
        }
        base_attrs.update(attrs)
        super().__init__(graph, base_attrs)

    @staticmethod
    def infer(node: Node):
        raise Exception('Not yet implemented')
        pass

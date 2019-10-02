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


import networkx as nx

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


# TODO: check all supported attributes in this file
class TensorIteratorInput(Op):
    op = "TensorIteratorInput"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'axis': None,
            'start': None,
            'end': None,
            'stride': None,
            'part_size': None,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': TensorIteratorInput.input_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return ['external_port_id', 'internal_layer_id', 'internal_port_id', 'axis', 'start', 'stride', 'part_size']

    @staticmethod
    def input_infer(node: Node):
        pass


class TensorIteratorOutput(Op):
    op = "TensorIteratorOutput"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'axis': None,
            'start': None,
            'end': None,
            'stride': None,
            'part_size': None,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': TensorIteratorOutput.input_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return ['external_port_id', 'internal_layer_id', 'internal_port_id', 'axis', 'start', 'stride', 'part_size']

    @staticmethod
    def input_infer(node: Node):
        pass


class TensorIteratorCondition(Op):
    op = "TensorIteratorCondition"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'in_ports_count': 2,
            'out_ports_count': 2,
            'infer': TensorIteratorCondition.input_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def input_infer(node: Node):
        pass


class TensorIteratorBackEdge(Op):
    op = 'TensorIteratorBackEdge'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': TensorIteratorBackEdge.input_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def input_infer(node: Node):
        pass

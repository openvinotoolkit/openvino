# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


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

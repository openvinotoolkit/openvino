# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer
from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.graph.perm_inputs import PermuteInputs
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class MVN(Op):
    op = 'MVN'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': self.op,
            'op': self.op,
            'version': 'opset6',
            'eps': None,
            'normalize_variance': None,
            'eps_mode': None,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.infer,
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
        }, attrs)

    def supported_attrs(self):
        return ['eps', 'eps_mode', 'normalize_variance']

    def backend_attrs(self):
        version = self.get_opset()
        if version == 'opset2':
            return ['eps',
                    ('across_channels', lambda node: bool_to_str(node, 'across_channels')),
                    ('normalize_variance', lambda node: bool_to_str(node, 'normalize_variance'))]
        elif version == 'opset6':
            return ['eps', 'eps_mode', ('normalize_variance', lambda node: bool_to_str(node, 'normalize_variance'))]
        else:
            raise Error('Unsupported MVN opset version "{}"'.format(version))

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        assert node.eps is not None, 'MVN required attribute `eps` unspecified for node {}'.format(name)
        assert node.normalize_variance is not None, \
            'MVN required attribute `normalize_variance` unspecified for node {}'.format(name)

        if node.version == 'opset6':
            assert node.eps_mode is not None, 'MVN required attribute `eps_mode` unspecified for node {}'.format(name)
            PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')

        copy_shape_infer(node)


class MVNOnnx(Op):
    op = 'MVNOnnx'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': None,
            'op': self.op,
            'version': None,
            'eps': None,
            'eps_mode': None,
            'normalize_variance': None,
            'axes': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': None
        }, attrs)


class MVNCaffe(Op):
    op = 'MVNCaffe'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': None,
            'op': self.op,
            'version': None,
            'eps': 1e-9,
            'normalize_variance': None,
            'across_channels': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': None
        }, attrs)

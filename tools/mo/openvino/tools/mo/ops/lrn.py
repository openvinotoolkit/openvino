# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class LRN(Op):
    op = 'LRN'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        assert 'alpha' in attrs, 'LRN operation should have `alpha` parameter set while creation'
        assert 'beta' in attrs, 'LRN operation should have `beta` parameter set while creation'
        assert 'bias' in attrs, 'LRN operation should have `bias` parameter set while creation'
        assert 'size' in attrs, 'LRN operation should have `size` parameter set while creation'
        assert 'region' not in attrs, \
            'LRN operation should not have `region` parameter set while creation, please use AttributedLRN operation ' \
            'instead or keep using LRN operation with region expressed as second `axis`-input'

        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',

            'infer': self.infer,
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['alpha', 'beta', 'bias', 'size']

    @staticmethod
    def infer(node):
        name = node.soft_get('name', node.id)

        connected_inputs = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_inputs) == 2 and 0 in connected_inputs and 1 in connected_inputs, \
            'LRN should have 2 connected input ports, but it doesn`t for node: `{}`. Ports: {}' \
            ''.format(name, connected_inputs)

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None, 'Input shape is unknown for node {}'.format(name)
        node.out_port(0).data.set_shape(input_shape)


class AttributedLRN(Op):
    op = 'AttributedLRN'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        assert 'alpha' in attrs, 'AttributedLRN operation should have `alpha` parameter set while creation'
        assert 'beta' in attrs, 'AttributedLRN operation should have `beta` parameter set while creation'
        assert 'local_size' in attrs, 'AttributedLRN operation should have `local_size` parameter set while creation'

        super().__init__(graph, {
            'op': self.op,
            'type': 'Norm',
            'version': 'opset1',

            'bias': 1,
            'region': 'across',
            'infer': self.infer,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

        assert 'region' in self.attrs, 'AttributedLRN operation should have `region` parameter set while creation'
        assert self.attrs['region'] in ['across', 'same'], \
            'AttributedLRN operation should have `region` parameter set to `across` or `same`, but it is `{}`' \
            ''.format(self.attrs['region'])

    def supported_attrs(self):
        return [
            'alpha',
            'beta',
            ('local-size', lambda node: node.local_size),
            'region'   # deprecated in V10 attribute, but it is kept for V6 compatibility
        ]

    @staticmethod
    def infer(node):
        name = node.soft_get('name', node.id)

        connected_inputs = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_inputs) == 1 and 0 in connected_inputs, \
            'AttributedLRN should have 1 connected input port, but it doesn`t for node: `{}`. Ports: {}' \
            ''.format(name, connected_inputs)

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None, 'Input shape is unknown for node {}'.format(name)
        node.out_port(0).data.set_shape(input_shape)

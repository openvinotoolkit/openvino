"""
 Copyright (C) 2018-2021 Intel Corporation

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
from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.layout import get_features_dim
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.extractor import bool_to_str
from mo.graph.graph import Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op
from mo.utils.error import Error


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
            'infer': self.infer
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
    def infer(node: None):
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

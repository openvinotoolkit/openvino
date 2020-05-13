"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph
from mo.ops.op import Op
from mo.utils.error import Error


class MVN(Op):
    op = 'MVN'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset2',
            'eps': None,
            'across_channels': 0,
            'normalize_variance': 1,
            'axes': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': __class__.infer
        }, attrs)

    def supported_attrs(self):
        return ['eps', 'across_channels', 'normalize_variance', 'axes']

    def backend_attrs(self):
        return ['eps', 'across_channels', 'normalize_variance']

    @staticmethod
    def infer(node: None):
        input_shape = node.in_node(0).shape
        name = node.soft_get('name', node.id)
        axes = node.axes
        if axes is not None:
            if 0 in axes:
                raise Error('Reduction over the batch dimension in node "{}" '
                            'is not supported by the backend.'.format(name))
            for i in range(2, len(input_shape)):
                if i not in axes:
                    raise Error(
                        'Reduction over spatial dimensions in node "{}" '
                        'is obligatory for the backend.'.format(name))
            if 1 in axes and not node.across_channels:
                raise Error('Inconsistent values of axes ({}) and across_channels ({}) parameters '
                            'in node "{}".'.format(str(axes), str(node.across_channels), name))

        copy_shape_infer(node)

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
            'across_channels': None,
            'normalize_variance': 1,
            'axes': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': __class__.infer
        }, attrs)

    def supported_attrs(self):
        return ['eps', 'across_channels', 'normalize_variance', 'axes']

    def backend_attrs(self):
        return ['eps',
                ('across_channels', lambda node: bool_to_str(node, 'across_channels')),
                ('normalize_variance', lambda node: bool_to_str(node, 'normalize_variance'))]

    @staticmethod
    def infer(node: None):
        input_shape = node.in_node(0).shape
        name = node.soft_get('name', node.id)

        if node.axes is not None and node.across_channels is not None:
            raise Error('Either axes or across_channels can be set for the MVN in node "{}".'.format(name))

        if node.across_channels is None:
            if node.axes is not None:
                # normalizing (replacing -1 with actual index)
                axes_data_value = node.axes
                axes = [axes_data_value.item()] if axes_data_value.size == 1 else axes_data_value
                axes = [get_canonical_axis_index(input_shape, a) for a in axes]
                # deduce across_channels from the axes, e.g. if the first axis is included (assuming batch is zero axis)
                feature_dim = get_features_dim(node.graph.graph['layout'], len(input_shape)) \
                    if (4 <= len(input_shape) <= 5) \
                    else 1
                node.across_channels = int(feature_dim in axes)

                if 0 in axes:
                    raise Error('Reduction over the batch dimension in node "{}" '
                                'is not supported by the backend.'.format(name))
                for i in range(2, len(input_shape)):
                    if i not in axes:
                        raise Error(
                            'Reduction over spatial dimensions in node "{}" '
                            'is obligatory for the backend.'.format(name))
            else:
                node.across_channels = 0  # default

        copy_shape_infer(node)

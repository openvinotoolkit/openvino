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
import numpy as np

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.op import Op


class SliceLike(Op):
    op = 'slice_like'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        assert 'axes' in attrs, 'Please set mandatory `axes` attribute for `slice_like` operation'
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.infer,
        }, attrs)

    @staticmethod
    def infer(node):
        input_shape = node.in_port(0).data.get_shape()
        shape_like = node.in_port(1).data.get_shape()

        new_shape = np.copy(input_shape)
        if node.axes is not None:
            node.axes = sorted([get_canonical_axis_index(input_shape, i) for i in node.axes])
            for i in node.axes:
                new_shape[i] = shape_like[i]
        else:
            assert input_shape.size == shape_like.size,\
                'Input shape ranks are inconsistent: {} and {}'.format(input_shape.size, shape_like.size)
            node.axes = int64_array(range(shape_like.size))
            new_shape = np.copy(shape_like)

        node.out_port(0).data.set_shape(new_shape)

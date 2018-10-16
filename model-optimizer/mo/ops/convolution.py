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

import networkx as nx

from mo.front.common.partial_infer.convolution import caffe_conv2d_infer
from mo.front.extractor import spatial_getter
from mo.ops.op import Op


class Convolution(Op):
    op = 'Convolution'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
            'infer': caffe_conv2d_infer
        }, attrs)

    def supported_attrs(self):
        return [
            'kernel',
            'pad',
            'stride',
            'output',
            'dilation'
        ]

    def backend_attrs(self):
        return [
            spatial_getter('stride-x', 'stride', 0),
            spatial_getter('stride-y', 'stride', 1),
            spatial_getter('kernel-x', 'kernel', 0),
            spatial_getter('kernel-y', 'kernel', 1),
            spatial_getter('dilation-x', 'dilation', 0),
            spatial_getter('dilation-y', 'dilation', 1),
            spatial_getter('pad-x', 'pad', 0, lambda x: x[0]),
            spatial_getter('pad-y', 'pad', 1, lambda x: x[0]),
            'output'
        ]

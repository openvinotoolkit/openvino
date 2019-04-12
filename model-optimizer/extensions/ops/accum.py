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

import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class AccumOp(Op):
    op = 'Accum'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'top_height': 0,
            'top_width': 0,
            'size_divisible_by': 0,
            'have_reference': 0,
            'out_ports_count': 1,
            'infer': AccumOp.accum_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'top_height',
            'top_width',
            'size_divisible_by',
            'have_reference'
        ]

    @staticmethod
    def accum_infer(node: Node):

        batch = node.in_node(0).shape[0]
        num_inputs = len(node.in_nodes())

        if node.have_reference:
            assert num_inputs >= 2, "Need at least two bottom blobs (one as reference)"
            total_channels = 0
            for i in range(num_inputs):
                total_channels += node.in_node(i).shape[1]
                assert node.in_node(i).shape[0] == batch, "All accumulated layers must have same number of images"
            assert total_channels >= 1, "Accumulated layers must have some channels in total"
            top_height_ = node.in_node(num_inputs - 1).shape[2]  # height
            top_width_ = node.in_node(num_inputs - 1).shape[3]  # width
            height_ = top_height_
            width_ = top_width_
        else:
            max_height = -1
            max_width = -1
            total_channels = 0

            for i in range(num_inputs):
                total_channels += node.in_node(i).shape[1]
                max_height = node.in_node(i).shape[2] if node.in_node(i).shape[2] > max_height else max_height
                max_width = node.in_node(i).shape[3] if node.in_node(i).shape[3] > max_width else max_width
                assert node.in_node(i).shape[0] == batch, "All accumulated layers must have same number of images"
            assert total_channels >= 1, "Accumulated layers must have some channels in total"

            if node.size_divisible_by:
                sdb = node.size_divisible_by
                top_height_ = int(np.ceil(max_height / sdb) * sdb)
                top_width_ = int(np.ceil(max_width / sdb) * sdb)
            else:
                top_height_ = node.top_height
                top_width_ = node.top_width
            if top_height_ > max_height and top_width_ > max_width:  # Layer can specify custom top size which is larger than default
                height_ = top_height_
                width_ = top_width_
            else:  # Otherwise maximum of bottom sizes will be used
                height_ = max_height
                width_ = max_width
        channels_ = total_channels
        node.out_node(0).shape = np.array([batch, channels_, height_, width_])

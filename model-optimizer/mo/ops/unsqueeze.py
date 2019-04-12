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

from mo.ops.op import Op, PermuteAttrs


class Unsqueeze(Op):
    op = 'Unsqueeze'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': 'Reshape',
            'op': __class__.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': __class__.infer
        }, attrs)

    @staticmethod
    def infer(node):
        unsqueeze_dims = np.array(node.unsqueeze_dims)
        value = node.in_node(0).value
        shape = node.in_node(0).shape

        for dim in unsqueeze_dims:
            shape = np.insert(shape, dim, 1)

        node.out_node().shape = np.array(shape)
        node['dim'] = shape
        PermuteAttrs.create_permute_attrs(node, attrs=[('dim', 'output:0')])

        if value is not None:
            value = np.reshape(value, shape)
            node.out_node().value = np.array(value)

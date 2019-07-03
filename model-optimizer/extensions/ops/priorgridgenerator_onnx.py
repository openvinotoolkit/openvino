"""
 Copyright (c) 2019 Intel Corporation

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

from mo.ops.op import Op


class ExperimentalDetectronPriorGridGenerator(Op):
    op = 'ExperimentalDetectronPriorGridGenerator'

    def __init__(self, graph, attrs):
        mandatory_props = dict(
            type=__class__.op,
            op=__class__.op,
            infer=__class__.infer,
        )
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return [
            'flatten',
            'h',
            'w',
            'stride_x',
            'stride_y',
        ]

    @staticmethod
    def infer(node):
        input_shape = node.in_node(0).shape
        priors_num = input_shape[0]
        grid_h = node.in_node(1).shape[2]
        grid_w = node.in_node(1).shape[3]
        if node.flatten:
            out_shape = np.array([grid_h * grid_w * priors_num, 4], dtype=np.int64)
        else:
            out_shape = np.array([grid_h, grid_w, priors_num, 4], dtype=np.int64)
        node.out_node(0).shape = out_shape

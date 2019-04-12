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


class ExperimentalDetectronTopKROIs(Op):
    op = 'ExperimentalDetectronTopKROIs'

    def __init__(self, graph, attrs):
        mandatory_props = dict(
            type=__class__.op,
            op=__class__.op,
            infer=__class__.infer
        )
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return ['max_rois', ]

    @staticmethod
    def infer(node):
        node.out_node(0).shape = np.array([node.max_rois, 4], dtype=np.int64)

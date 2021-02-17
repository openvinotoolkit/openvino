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

import numpy as np

from extensions.ops.mvn import MVN
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.const import Const


class MeanVarianceNormalizationExtractor(FrontExtractorOp):
    op = 'MeanVarianceNormalization'
    enabled = True

    @classmethod
    def extract(cls, node):
        name = node.soft_get('name', node.id)
        axes = onnx_attr(node, 'axes', 'ints',
                         default=np.array([0, 2, 3], dtype=np.int64),
                         dst_type=lambda x: np.array(x, dtype=np.int64))

        axes = Const(node.graph, {'value': axes, 'name': name + '/Axes'}).create_node()
        node.add_input_port(1, skip_if_exist=True)
        node.in_port(1).connect(axes.out_port(0))

        attrs = {
            'eps': 1e-9,
            'normalize_variance': 1,
            'eps_mode': 'outside_sqrt'
        }

        MVN.update_node_stat(node, attrs)
        return cls.enabled

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

from extensions.ops.mvn import MVN
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.utils.error import Error


class MeanVarianceNormalizationExtractor(FrontExtractorOp):
    op = 'MeanVarianceNormalization'
    enabled = True

    @classmethod
    def extract(cls, node):
        name = node.soft_get('name', node.id)
        axes = onnx_attr(node, 'axes', 'ints',
                         default=np.array([0, 2, 3], dtype=np.int64),
                         dst_type=lambda x: np.array(x, dtype=np.int64))

        if axes is not None:
            if 0 in axes:
                raise Error('Reduction over the batch dimension in node "{}" '
                            'is not supported by the backend.'.format(name))
            # Dimension 4 (if it's present in the input tensor)
            # should also be in the list of axes for reduction.
            # This case will be handled at the MVN Op side,
            # 'cause input shape is not available at that stage.
            for i in (2, 3):
                if i not in axes:
                    raise Error(
                        'Reduction over spatial dimensions in node "{}" '
                        'is obligatory for the backend.'.format(name))

        attrs = {
            'eps': 1e-9,
            'across_channels': 1 if 1 in axes else 0,
            'normalize_variance': 1,
            'axes': axes
        }
        MVN.update_node_stat(node, attrs)
        return cls.enabled

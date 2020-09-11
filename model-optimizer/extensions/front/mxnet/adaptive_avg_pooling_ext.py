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

from extensions.ops.adaptive_avg_pooling import AdaptiveAvgPooling
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class AdaptiveAvgPooling2DFrontExtractor(FrontExtractorOp):
    op = '_contrib_AdaptiveAvgPooling2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        output_size = attrs.tuple("output_size", int, None)
        if len(output_size) == 1:
            output_size = (output_size[0], output_size[0])

        data = {
            'op': 'Pooling',
            'output_size': output_size
        }
        AdaptiveAvgPooling.update_node_stat(node, data)
        return cls.enabled
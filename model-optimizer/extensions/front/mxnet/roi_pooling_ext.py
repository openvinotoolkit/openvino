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

from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.ops.roipooling import ROIPooling


class ROIPoolingFrontExtractor(FrontExtractorOp):
    op = 'ROIPooling'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)

        spatial_scale = attrs.float("spatial_scale", None)
        pooled_size = attrs.tuple("pooled_size", int, (0, 0))
        data = {
            'type': 'ROIPooling',
            'spatial_scale': spatial_scale,
            'pooled_w': pooled_size[1],
            'pooled_h': pooled_size[0]
        }

        data.update(layout_attrs())

        # update the attributes of the node
        ROIPooling.update_node_stat(node, data)
        return __class__.enabled
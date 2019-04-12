"""
 Copyright (c) 2017-2019 Intel Corporation

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

from mo.graph.graph import Node
from extensions.ops.instance_normalization import InstanceNormalization
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class InstanceNormFrontExtractor(FrontExtractorOp):
    op = 'InstanceNorm'
    enabled = True

    @staticmethod
    def extract(node: Node):
        attr = get_mxnet_layer_attrs(node.symbol_dict)
        node_attrs = {
            'epsilon': attr.float('eps', 0.001)
        }

        InstanceNormalization.update_node_stat(node, node_attrs)
        return __class__.enabled

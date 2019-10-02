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

from extensions.ops.MatMul import FullyConnected
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class FullyConnectedFrontExtractor(FrontExtractorOp):
    op = 'FullyConnected'
    enabled = True

    @classmethod
    def extract(cls, node):
        attr = get_mxnet_layer_attrs(node.symbol_dict)
        num_hidden = attr.int('num_hidden', None)
        assert num_hidden is not None, "{} node with no `num_hidden` parameter found".format(cls.op)
        attrs = {
            'out-size': num_hidden,
            'transpose_weights': True,
        }
        FullyConnected.update_node_stat(node, attrs)
        return cls.enabled

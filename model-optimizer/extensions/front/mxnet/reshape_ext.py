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

from extensions.ops.mxreshape import MXReshape
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.ops.reshape import Reshape


class ReshapeFrontExtractor(FrontExtractorOp):
    op = 'Reshape'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        dim = attrs.tuple("shape", int, None)
        reverse = attrs.bool("reverse", False)
        update_attrs = {
            'dim': int64_array(dim),
            'reverse': reverse
        }
        for d in dim:
            if d in [-2, -3, -4] or reverse:
                MXReshape.update_node_stat(node, update_attrs)
                return cls.enabled

        # update the attributes of the node
        Reshape.update_node_stat(node, update_attrs)
        return cls.enabled

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

import logging as log
import numpy as np

from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.front.extractor import FrontExtractorOp
from mo.ops.reshape import Reshape


class ReshapeFrontExtractor(FrontExtractorOp):
    op = 'Reshape'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        dim = attrs.tuple("shape", int, None)
        update_attrs = {
            'dim': np.array(dim)
        }
        for d in dim:
            if d in [-2, -3, -4]:
                log.error('The attribute "shape" of the operation "{}" contains value "{}" which is not supported.'.
                          format(node.soft_get('name'), d))
                return False

        # update the attributes of the node
        Reshape.update_node_stat(node, update_attrs)
        return __class__.enabled

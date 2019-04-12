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

from mo.front.extractor import FrontExtractorOp, MXNetCustomFrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class CustomFrontExtractorOp(FrontExtractorOp):
    op = 'Custom'
    enabled = True

    def extract(self, node):
        supported = False
        op_attrs = None
        node_attrs = get_mxnet_layer_attrs(node.symbol_dict)
        op_type = node_attrs.str('op_type', None)
        if op_type and op_type in MXNetCustomFrontExtractorOp.registered_ops:
            supported, op_attrs = MXNetCustomFrontExtractorOp.registered_ops[op_type]().extract(node)
        return supported, op_attrs

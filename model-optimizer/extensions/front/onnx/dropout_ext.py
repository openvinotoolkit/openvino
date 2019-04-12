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

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from extensions.ops.identity import IdentityOp
from mo.utils.error import Error


class DropoutFrontExtractor(FrontExtractorOp):
    op = 'Dropout'
    enabled = True

    @staticmethod
    def extract(node):
        # some Dropout flavors doesn't have is_test attribute; when it is missing, interpret it as 1
        is_test = onnx_attr(node, 'is_test', 'i', 1)
        if len(node.out_nodes()) > 1:
            raise Error('Dropout node {} has more than one consumer. Unsupported.', node.name)
        if not is_test:
            raise Error('Dropout node {} has is_test: 0. This means training mode which is not supported.', node.name)
        IdentityOp.update_node_stat(node)
        return __class__.enabled

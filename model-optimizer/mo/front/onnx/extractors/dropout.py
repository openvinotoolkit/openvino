"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.onnx.extractors.utils import onnx_attr
from mo.utils.error import Error

def dropout_ext(node):
    # some Dropout flavors doesn't have is_test attribute; when it is missing, interpret it as 1
    is_test = onnx_attr(node, 'is_test', 'i', 1)
    if len(node.out_nodes()) > 1:
        raise Error('Dropout node {} has more than one consumer. Unsupported.', node.name)
    if not is_test:
        raise Error('Dropout node {} has is_test: 0. This means training mode which is not supported.', node.name)

    return {
        # redefine op to automatically remove a node in the next tranformations
        'op': 'Identity',
    }


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

from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from mo.ops.unsqueeze import Unsqueeze


class ExpandDimsExtractor(FrontExtractorOp):
    """
    Due to historical reasons the ExpandDims operation in the Model Optimizer has one input with data and the attribute
    which specifies the dimension to expand. But in the TensorFlow the ExpandDims operation has 2 inputs where the
    second input specifies the dimensions to expand. In the Model Optimizer this operation corresponds to the Unsqueeze.
    """
    op = 'ExpandDims'
    enabled = True

    @staticmethod
    def extract(node: Node):
        Unsqueeze.update_node_stat(node, {})
        return __class__.enabled

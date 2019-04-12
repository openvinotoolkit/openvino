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

from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from mo.ops.lin_op import Add


class BiasToAdd(FrontExtractorOp):
    """
    Replaces Bias layer with Eltwise.
    """
    op = "Bias"
    enabled = True

    @staticmethod
    def extract(node: Node):
        attrs = {'axis': node.pb.bias_param.axis}
        embed_input(attrs, 1, 'bias', node.model_pb.blobs[0].data, 'biases')

        Add.update_node_stat(node, attrs)

        return __class__.enabled

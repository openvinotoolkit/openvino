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

from mo.front.caffe.extractors.utils import weights_biases
from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op


class AffineTransformFrontExtractor(FrontExtractorOp):
    op = 'affinetransform'
    enabled = True

    @staticmethod
    def extract(node):
        mapping_rule = {
            'out-size': node.pb.num_output,
            'layout': 'NCHW'
        }
        mapping_rule.update(weights_biases(node.pb.bias_term, node.pb))

        Op.get_op_class_by_name('FullyConnected').update_node_stat(node, mapping_rule)
        return __class__.enabled

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

from mo.front.caffe.collect_attributes import collect_attributes
from mo.front.extractor import FrontExtractorOp
from extensions.ops.activation_ops import Elu


class ELUFrontExtractor(FrontExtractorOp):
    op = 'ELU'
    enabled = True

    @staticmethod
    def extract(node):
        param = node.pb.elu_param
        attrs = collect_attributes(param)

        Elu.update_node_stat(node, attrs)
        return __class__.enabled

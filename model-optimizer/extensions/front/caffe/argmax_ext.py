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
from extensions.ops.argmax import ArgMaxOp
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.extractor import FrontExtractorOp


class ArgMaxFrontExtractor(FrontExtractorOp):
    op = 'ArgMax'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.argmax_param

        update_attrs = {
            'out_max_val': int(param.out_max_val),
            'top_k': param.top_k,
            'axis': param.axis,
        }

        mapping_rule = merge_attrs(param, update_attrs)

        ArgMaxOp.update_node_stat(node, mapping_rule)
        # ArgMax must be converted to TopK but without the output with values
        ArgMaxOp.update_node_stat(node, {'remove_values_output': True})
        return cls.enabled

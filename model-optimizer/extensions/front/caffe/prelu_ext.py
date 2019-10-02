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
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.caffe.extractors.utils import weights_biases
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op


class PreluFrontExtractor(FrontExtractorOp):
    op = 'PReLU'
    enabled = True

    @staticmethod
    def extract(node):
        proto_layer = node.pb
        pb_model = node.model_pb
        param = proto_layer.prelu_param

        update_attrs = {
            'channel_shared': int(param.channel_shared)
        }

        variance_norm_caffe_map = {
            0: 'caffe.FillerParameter.FAN_IN',
            1: 'caffe.FillerParameter.FAN_OUT',
            2: 'caffe.FillerParameter.AVERAGE'
        }

        if hasattr(param, 'filler'):
            update_attrs.update({
                'filler_type': param.filler.type,
                'filler_value': int(param.filler.value),
                'min': int(param.filler.min),
                'max': int(param.filler.max),
                'mean': int(param.filler.mean),
                'std': int(param.filler.std),
                'sparse': param.filler.sparse,
                'variance_norm': variance_norm_caffe_map[param.filler.variance_norm]
            })

        mapping_rule = merge_attrs(param, update_attrs)
        mapping_rule.update(weights_biases(False, pb_model))
        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        Op.get_op_class_by_name(__class__.op).update_node_stat(node, mapping_rule)
        return __class__.enabled

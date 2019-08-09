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
from extensions.ops.interpolate import Interpolate
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp


class ResampleFrontExtractor(FrontExtractorOp):
    op = 'Resample'
    enabled = True

    @staticmethod
    def extract(node):
        proto_layer = node.pb
        param = proto_layer.resample_param
        types = [
            "",
            'nearest',
            'linear',
            'cubic',
            'area',
        ]
        resample_type = types[param.type]

        update_attrs = {
            'antialias': int(param.antialias),
            'height': param.height,
            'width': param.width,
            'type': resample_type,
            'factor': param.factor,
            'fw': 'caffe',
        }

        mapping_rule = merge_attrs(param, update_attrs)
        mapping_rule['mode'] = mapping_rule['type']
        mapping_rule['axes'] = int64_array([2, 3])
        mapping_rule.pop('type')
        Interpolate.update_node_stat(node, mapping_rule)
        return __class__.enabled

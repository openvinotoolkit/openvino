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
import numpy as np

from mo.front.caffe.collect_attributes import merge_attrs, collect_attributes
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op


class RegionYoloFrontExtractor(FrontExtractorOp):
    op = 'RegionYolo'
    enabled = True

    @staticmethod
    def extract(node):
        proto_layer = node.pb
        param = proto_layer.region_yolo_param
        flatten_param = proto_layer.flatten_param
        axis = flatten_param.axis
        end_axis = flatten_param.end_axis
        coords = param.coords
        classes = param.classes
        num = param.num
        update_attrs = {
            'coords': coords,
            'classes': classes,
            'num': num,
            'do_softmax': int(param.do_softmax),
            'anchors': np.array(param.anchors),
            'mask': np.array(param.mask)
        }

        flatten_attrs = {
            'axis': axis,
            'end_axis': end_axis
        }

        mapping_rule = merge_attrs(param, update_attrs)

        mapping_rule.update(flatten_attrs)
        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        Op.get_op_class_by_name(__class__.op).update_node_stat(node, mapping_rule)
        return __class__.enabled

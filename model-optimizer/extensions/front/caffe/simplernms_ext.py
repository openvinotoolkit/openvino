# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.simplernms import SimplerNMSOp
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.extractor import FrontExtractorOp


class SimplerNMSFrontExtractor(FrontExtractorOp):
    op = 'SimplerNMS'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.simpler_nms_param
        update_attrs = {
            'cls_threshold': param.cls_threshold,
            'max_num_proposals': param.max_num_proposals,
            'iou_threshold': param.iou_threshold,
            'min_bbox_size': param.min_bbox_size,
            'feat_stride': param.feat_stride,
            'pre_nms_topn': param.pre_nms_topn,
            'post_nms_topn': param.post_nms_topn,
            'scale': param.scale,
        }

        mapping_rule = merge_attrs(param, update_attrs)

        # update the attributes of the node
        SimplerNMSOp.update_node_stat(node, mapping_rule)
        return cls.enabled

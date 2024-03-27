# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.DetectionOutput import DetectionOutput
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class MultiBoxDetectionOutputExtractor(FrontExtractorOp):
    op = '_contrib_MultiBoxDetection'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        top_k = attrs.int("nms_topk", -1)
        nms_threshold = attrs.float("nms_threshold", 0.5)
        confidence_threshold = attrs.float("threshold", 0.01)
        clip = 0 if not attrs.bool("clip", True) else 1

        node_attrs = {
            'type': 'DetectionOutput',
            'op': __class__.op,
            'keep_top_k': top_k,
            'variance_encoded_in_target': 0,
            'code_type': "caffe.PriorBoxParameter.CENTER_SIZE",
            'share_location': 1,
            'confidence_threshold': confidence_threshold,
            'background_label_id': 0,
            'nms_threshold': nms_threshold,
            'top_k': top_k,
            'decrease_label_id': 1,
            'clip_before_nms': clip,
            'normalized': 1,
        }

        DetectionOutput.update_node_stat(node, node_attrs)

        return cls.enabled

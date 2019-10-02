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
from mo.front.common.partial_infer.multi_box_detection import multi_box_detection_infer


def multi_box_detection_ext(attrs):
    num_classes = 21
    top_k = attrs.int("nms_topk", -1)
    keep_top_k = top_k
    variance_encoded_in_target = 0
    code_type = "caffe.PriorBoxParameter.CENTER_SIZE"
    share_location = 1
    nms_threshold = attrs.float("nms_threshold", 0.5)
    confidence_threshold = attrs.float("threshold", 0.01)
    background_label_id = 0
    clip = 0 if not attrs.bool("clip", True) else 1
    node_attrs = {
        'type': 'DetectionOutput',
        'num_classes': num_classes,
        'keep_top_k': keep_top_k,
        'variance_encoded_in_target': variance_encoded_in_target,
        'code_type': code_type,
        'share_location': share_location,
        'confidence_threshold': confidence_threshold,
        'background_label_id': background_label_id,
        'nms_threshold': nms_threshold,
        'top_k': top_k,
        'decrease_label_id': 1,
        'clip': clip,
        'normalized': 1,
        'infer': multi_box_detection_infer
    }
    return node_attrs

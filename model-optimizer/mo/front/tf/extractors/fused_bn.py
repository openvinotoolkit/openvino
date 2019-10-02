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
import logging as log

import numpy as np

from mo.front.tf.extractors.utils import tf_dtype_extractor


def tf_fused_bn_infer(node):
    output_shape = np.array(node.in_node(0).shape)
    for port, out_node in node.out_nodes().items():
        out_node.shape = output_shape


def tf_fused_bn_extractor(pb):
    is_training = pb.attr['is_training'].b
    if is_training:
        log.warning('FusedBatchNorm doesn\'t support is_training=True')

    return {
        'data_format': pb.attr["data_format"].s,
        'data_type': tf_dtype_extractor(pb.attr["T"].type),
        'eps': pb.attr['epsilon'].f,
        'infer': tf_fused_bn_infer,
        'is_training': is_training
    }

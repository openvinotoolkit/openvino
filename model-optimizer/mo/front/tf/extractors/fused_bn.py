# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

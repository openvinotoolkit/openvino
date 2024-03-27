# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.random_uniform import RandomUniform
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor


class RandomUniformIntExtractor(FrontExtractorOp):
    op = 'RandomUniformInt'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'output_type': tf_dtype_extractor(node.pb.attr["Tout"].type),
            'global_seed': node.pb.attr['seed'].i,
            'op_seed': node.pb.attr['seed2'].i
        }
        RandomUniform.update_node_stat(node, attrs)
        return cls.enabled

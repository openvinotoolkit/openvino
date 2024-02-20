# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor, tf_tensor_shape, tf_tensor_content
from openvino.tools.mo.ops.const import Const


class ConstExtractor(FrontExtractorOp):
    op = 'Const'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb_tensor = node.pb.attr["value"].tensor
        shape = tf_tensor_shape(pb_tensor.tensor_shape)
        attrs = {
            'shape': shape,
            'value': tf_tensor_content(pb_tensor.dtype, shape, pb_tensor),
            'data_type': tf_dtype_extractor(pb_tensor.dtype),
        }
        Const.update_node_stat(node, attrs)
        return cls.enabled

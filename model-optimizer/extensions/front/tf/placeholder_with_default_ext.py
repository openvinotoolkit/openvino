# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer, copy_value
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_dtype_extractor, tf_tensor_shape
from mo.ops.op import Op


class PlaceholderWithDefaultExtractor(FrontExtractorOp):
    op = 'PlaceholderWithDefault'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'data_type': tf_dtype_extractor(node.pb.attr["dtype"].type),
            'shape': tf_tensor_shape(node.pb.attr["shape"].shape),
            'identity': True,
            'infer': lambda node: copy_shape_infer(node, value_infer=copy_value),
        }
        Op.update_node_stat(node, attrs)
        return cls.enabled

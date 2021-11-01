# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor, tf_tensor_shape
from openvino.tools.mo.ops.op import PermuteAttrs


class PlaceholderFrontExtractor(FrontExtractorOp):
    op = 'Placeholder'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'data_type': tf_dtype_extractor(node.pb.attr["dtype"].type),
            'shape': tf_tensor_shape(node.pb.attr["shape"].shape),
            'permute_attrs': PermuteAttrs().update_attrs(attrs=[('shape', 'output:0')])
        }
        Parameter.update_node_stat(node, attrs)
        return cls.enabled

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.TensorArray import TensorArray
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_tensor_shape
from mo.graph.graph import Node


class TensorArrayExtractor(FrontExtractorOp):
    op = "TensorArrayV3"
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = {
            'op': __class__.op,
            'element_shape': tf_tensor_shape(node.pb.attr["element_shape"].shape),
        }
        TensorArray.update_node_stat(node, attrs)
        return cls.enabled
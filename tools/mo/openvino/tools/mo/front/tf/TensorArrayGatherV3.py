# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.TensorArrayGather import TensorArrayGather
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_tensor_shape
from openvino.tools.mo.graph.graph import Node


class TensorArrayGatherV3Extractor(FrontExtractorOp):
    op = "TensorArrayGatherV3"
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = {
            'op': __class__.op,
            'element_shape': tf_tensor_shape(node.pb.attr["element_shape"].shape),
        }
        TensorArrayGather.update_node_stat(node, attrs)
        return cls.enabled


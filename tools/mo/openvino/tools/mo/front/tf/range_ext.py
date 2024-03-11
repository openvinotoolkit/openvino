# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor
from openvino.tools.mo.graph.graph import Node


class RangeFrontExtractor(FrontExtractorOp):
    op = 'Range'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Range.update_node_stat(node, {'output_type': tf_dtype_extractor(node.pb.attr['Tidx'].type)})
        return cls.enabled


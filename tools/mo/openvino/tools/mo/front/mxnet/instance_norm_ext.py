# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.instance_normalization import InstanceNormalization
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.graph.graph import Node


class InstanceNormFrontExtractor(FrontExtractorOp):
    op = 'InstanceNorm'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attr = get_mxnet_layer_attrs(node.symbol_dict)
        node_attrs = {
            'epsilon': attr.float('eps', 0.001)
        }

        InstanceNormalization.update_node_stat(node, node_attrs)
        return cls.enabled

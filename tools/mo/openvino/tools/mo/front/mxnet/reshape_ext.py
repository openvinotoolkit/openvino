# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.mxreshape import MXReshape
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.reshape import Reshape


class ReshapeFrontExtractor(FrontExtractorOp):
    op = 'Reshape'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        dim = attrs.tuple("shape", int, None)
        reverse = attrs.bool("reverse", False)
        update_attrs = {
            'dim': int64_array(dim),
            'reverse': reverse
        }
        for d in dim:
            if d in [-2, -3, -4] or reverse:
                MXReshape.update_node_stat(node, update_attrs)
                return cls.enabled

        # update the attributes of the node
        Reshape.update_node_stat(node, update_attrs)
        return cls.enabled

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.swapaxis import SwapAxis
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


def extract(node):
    attrs = get_mxnet_layer_attrs(node.symbol_dict)
    dim1 = attrs.int("dim1", 0)
    dim2 = attrs.int("dim2", 0)

    update_attrs = {
        'dim1': dim1,
        'dim2': dim2,
    }

    # update the attributes of the node
    SwapAxis.update_node_stat(node, update_attrs)
    return True


class SwapAxisFrontExtractor(FrontExtractorOp):
    op = 'SwapAxis'
    enabled = True

    extract = staticmethod(extract)


class SwapAxesFrontExtractor(FrontExtractorOp):
    op = 'swapaxes'
    enabled = True

    extract = staticmethod(extract)

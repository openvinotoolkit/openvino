# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.ops.crop import Crop


class CropFrontExtractor(FrontExtractorOp):
    op = 'Crop'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        offset = attrs.tuple("offset", int, ())
        axis = attrs.int("num_args", 0)
        node_attrs = {
            'axis': axis,
            'offset': list(offset),
            'dim': None,
        }
        Crop.update_node_stat(node, node_attrs)
        return cls.enabled


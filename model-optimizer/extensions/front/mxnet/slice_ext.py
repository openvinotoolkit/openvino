# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import mo_array
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.ops.slice import MXSlice


class SliceFrontExtractor(FrontExtractorOp):
    op = 'slice'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node_attrs = {
            'crop_begin': mo_array(attrs.tuple("begin", int, ())),
            'crop_end': mo_array(attrs.tuple("end", int, ())),
            'step': mo_array(attrs.tuple("step", int, ())),
        }

        MXSlice.update_node_stat(node, node_attrs)
        return cls.enabled

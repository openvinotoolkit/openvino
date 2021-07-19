# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.roll import AttributedRoll
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class RandomUniformExtractor(FrontExtractorOp):
    op = '_random_uniform'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        shape = list(attrs.tuple("shape", int, None))
        new_attrs = {'shape': shape}
        if attrs.has("high"):
            high = attrs.tuple("high", float, 1.0)
            new_attrs.update({'max_val': high})
        if attrs.has("low"):
            low = attrs.tuple("low", float, 0.0)
            new_attrs.update({'min_val': low})
        if attrs.has("dtype"):
            low = attrs.tuple("dtype", str, 'float32')
            new_attrs.update({'min_val': low})
        AttributedRoll.update_node_stat(node, new_attrs)
        return cls.enabled

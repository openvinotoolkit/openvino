# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.MatMul import FullyConnected
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class FullyConnectedFrontExtractor(FrontExtractorOp):
    op = 'FullyConnected'
    enabled = True

    @classmethod
    def extract(cls, node):
        attr = get_mxnet_layer_attrs(node.symbol_dict)
        num_hidden = attr.int('num_hidden', None)
        assert num_hidden is not None, "{} node with no `num_hidden` parameter found".format(cls.op)
        attrs = {
            'out-size': num_hidden,
            'transpose_weights': True,
        }
        FullyConnected.update_node_stat(node, attrs)
        return cls.enabled

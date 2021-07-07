# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.graph.graph import Node
from mo.ops.tile import Tile


class TileExt(FrontExtractorOp):
    op = 'tile'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        Tile.update_node_stat(node, {
            'reps': attrs.tuple('reps', int, None),
        })
        return cls.enabled

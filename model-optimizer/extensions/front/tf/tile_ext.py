# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.ops.tile import Tile


class TileExtractor(FrontExtractorOp):
    op = 'Tile'
    enabled = True

    @classmethod
    def extract(cls, node):
        Tile.update_node_stat(node, {})
        return cls.enabled

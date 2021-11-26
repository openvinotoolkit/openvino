# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.ops.space_to_batch import BatchToSpace


class SpaceToBatchFrontExtractor(FrontExtractorOp):
    op = 'BatchToSpaceND'
    enabled = True

    @classmethod
    def extract(cls, node):
        BatchToSpace.update_node_stat(node)
        return cls.enabled

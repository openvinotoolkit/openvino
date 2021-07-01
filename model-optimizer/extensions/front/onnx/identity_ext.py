# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from extensions.ops.identity import Identity


class IdentityFrontExtractor(FrontExtractorOp):
    op = 'Identity'
    enabled = True

    @classmethod
    def extract(cls, node):
        Identity.update_node_stat(node)
        return cls.enabled

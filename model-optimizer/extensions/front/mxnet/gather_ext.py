# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp


class GatherFrontExtractor(FrontExtractorOp):
    op = 'Embedding'
    enabled = True

    @classmethod
    def extract(cls, node):
        return cls.enabled

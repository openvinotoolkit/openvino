# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.replacement import FrontReplacementOp, FrontReplacementPattern, FrontReplacementSubgraph
from mo.front.extractor import FrontExtractorOp, CaffePythonFrontExtractorOp


def get_front_classes():
    front_classes = [FrontExtractorOp, CaffePythonFrontExtractorOp, FrontReplacementOp,
                     FrontReplacementPattern, FrontReplacementSubgraph]
    return front_classes

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementOp, FrontReplacementSubgraph, FrontReplacementPattern
from openvino.tools.mo.front.extractor import FrontExtractorOp


def get_front_classes():
    front_classes = [FrontExtractorOp, FrontReplacementOp, FrontReplacementPattern, FrontReplacementSubgraph]
    return front_classes

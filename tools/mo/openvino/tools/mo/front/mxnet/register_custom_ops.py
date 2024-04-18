# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementOp, FrontReplacementSubgraph, FrontReplacementPattern
from openvino.tools.mo.front.extractor import FrontExtractorOp, MXNetCustomFrontExtractorOp
from openvino.tools.mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral

def get_front_classes():
    front_classes = [FrontExtractorOp, FrontReplacementOp, FrontReplacementSubgraph, MXNetCustomFrontExtractorOp,
                     FrontReplacementPattern, FrontReplacementFromConfigFileGeneral]
    return front_classes

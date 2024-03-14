# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementOp, FrontReplacementPattern, FrontReplacementSubgraph
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph, FrontReplacementFromConfigFileOp, \
    FrontReplacementFromConfigFileGeneral


def get_front_classes():
    front_classes = [FrontExtractorOp, FrontReplacementOp, FrontReplacementPattern, FrontReplacementSubgraph,
                     FrontReplacementFromConfigFileSubGraph, FrontReplacementFromConfigFileOp,
                     FrontReplacementFromConfigFileGeneral]
    return front_classes

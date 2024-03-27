# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.front.common.partial_infer.utils import mark_input_bins
from openvino.tools.mo.graph.graph import Node


def batch_norm_4_infer(node: Node):
    copy_shape_infer(node)
    mark_input_bins(node, ['weights', 'biases', 'mean', 'variance'])
    if node.has('fix_gamma') and node.fix_gamma:
        # go to the 1-st input weights and set all elements to 1
        node.in_node(1).value = np.full_like(node.in_node(1).value, 1, dtype=np.float32)

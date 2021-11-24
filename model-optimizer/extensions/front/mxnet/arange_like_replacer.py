# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.gather import Gather
from extensions.ops.range import Range
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph
from mo.ops.shape import Shape
from mo.ops.reshape import Reshape


class ArangeLikeReplacer(FrontReplacementOp):
    op = 'arange_like'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        pass

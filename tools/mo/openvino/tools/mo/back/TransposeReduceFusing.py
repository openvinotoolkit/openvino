# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import numpy as np

from openvino.tools.mo.back.FuseTransposesSequence import FuseTransposesSequence
from openvino.tools.mo.back.ReduceMerge import ReduceMerge
from openvino.tools.mo.ops.ReduceOps import reduce_map
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node


class TransposeReduce(BackReplacementPattern):
    """
    Fuse Transpose--->Reduce to Reduce with correct reduce axis input
    """
    # TODO: Make another implementation, this is a temporary solution for one case
    enabled = True
    force_clean_up = True

    def run_before(self):
        return [ReduceMerge]

    def run_after(self):
        return [FuseTransposesSequence]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('transpose_const', dict(kind='op', type='Const', value=lambda v: v is not None and
                                                                      np.array_equal(v, int64_array([0, 2, 3, 1])))),
                ('transpose_const_data', dict(kind='data')),
                ('transpose', dict(kind='op', type='Transpose')),
                ('transpose_data', dict(kind='data')),
                ('reduce_const', dict(kind='op', type='Const', value=lambda v: v is not None and
                                                                       np.array_equal(v, int64_array([1, 2])))),
                ('reduce_const_data', dict(kind='data')),
                ('reduce', dict(kind='op', type=lambda t: t in reduce_map.keys(), keep_dims=False))
            ],
            edges=[
                ('transpose_const', 'transpose_const_data'),
                ('transpose_const_data', 'transpose', {'in': 1}),
                ('transpose', 'transpose_data'),
                ('transpose_data', 'reduce', {'in': 0}),
                ('reduce_const', 'reduce_const_data'),
                ('reduce_const_data', 'reduce', {'in': 1})
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: Dict[str, Node]):
        transpose = match['transpose']
        reduce = match['reduce']
        gather = create_op_with_const_inputs(graph, op=Gather, port_value_dict={2: int64_array(0)},
                                             op_attrs={'name': reduce.name + 'Gather'})

        transpose.in_port(1).get_connection().set_destination(gather.in_port(0))
        reduce.in_port(1).get_connection().set_destination(gather.in_port(1))

        gather.out_port(0).connect(reduce.in_port(1))
        transpose.out_port(0).disconnect()
        transpose.in_port(0).get_connection().set_destination(reduce.in_port(0))

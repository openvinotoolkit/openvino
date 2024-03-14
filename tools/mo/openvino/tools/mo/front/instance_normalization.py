# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Add, Mul
from openvino.tools.mo.ops.mvn import MVN
from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.ops.rank import Rank
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Node, Graph, rename_nodes

import numpy as np


class InstanceNormalization(FrontReplacementOp):
    ''' Decompose InstanceNormalization to scale*MVN(x) + B

        There are should be also reshapes added for each scale and B.
    '''
    op = "InstanceNormalization"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        name = node.soft_get('name', node.id)

        # create range of axes for MVN based on `start_axis` and rank of input
        rank = Rank(graph, {'name': name + '/Rank'}).create_node()
        rng = create_op_with_const_inputs(graph, Range, {0: int64_array(2), 2: int64_array(1)},
                                          {'name': name + '/Range', 'output_type': np.int64})
        mvn = MVN(graph, {'eps': node.epsilon, 'eps_mode': 'inside_sqrt', 'normalize_variance': 1,
                          'name': name + '/Ins_Norm/MVN_', }).create_node()
        node.in_port(0).get_connection().set_destination(mvn.in_port(0))
        rng.out_port(0).connect(mvn.in_port(1))
        mul = Mul(graph, {'axis': 1, 'name': name + '/Ins_Norm/mul_'}).create_node()
        mvn.out_port(0).connect(mul.in_port(0))
        node.in_port(1).get_connection().set_destination(mul.in_port(1))
        add = Add(graph, {'axis': 1, 'name': name + '/Ins_Norm/add_'}).create_node()
        mul.out_port(0).connect(add.in_port(0))
        node.in_port(2).get_connection().set_destination(add.in_port(1))

        mvn.in_port(0).get_connection().add_destination(rank.in_port(0))
        rng.in_port(1).connect(rank.out_port(0))

        rename_nodes([(node, name + '/TBD'), (add, name)])

        return [add.id]

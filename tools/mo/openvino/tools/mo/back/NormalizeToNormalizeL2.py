# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.back.insert_compatibility_l2normalization import CompatibilityL2NormalizationPattern
from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.ops.normalize_l2 import NormalizeL2Op
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_node


class NormalizeToNormalizeL2(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [CompatibilityL2NormalizationPattern]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('normalize', {'type': 'Normalize'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['normalize']

        # rename normalize node since it will be no longer output node after the transformation
        output_name = node.soft_get('name', node.id)
        normalizel2_name = output_name + '/normalizel2'
        rename_node(node, normalizel2_name)

        assert node.in_port(0).data.get_shape().size in [2, 3, 4]
        assert node.has_valid('across_spatial')
        assert node.has_valid('channel_shared')
        assert node.has_valid('eps')

        if 'bin' in node.in_edge(1):
            del node.in_edge(1)['bin']

        weights = node.in_port(1).data.get_value()
        assert weights is not None
        # in the code below we intentionally use get_source() to get the out port. Because updating the out port will
        # update the Const node 'value' and 'shape' attributes
        if node.channel_shared or all(weights == weights[0]):
            node.in_port(1).get_source().data.set_value(mo_array([weights[0]]))
        else:
            new_shape = np.ones((len(node.in_port(0).data.get_shape())), dtype=np.int64)
            new_shape[1] = -1
            node.in_port(1).get_source().data.set_value(mo_array(weights).reshape(new_shape))

        mul = Mul(graph, {'name': output_name}).create_node()
        rename_node(mul, output_name)

        if not node.across_spatial:
            axes = int64_array([1])
        else:
            axes = int64_array(np.arange(start=1, stop=node.in_port(0).data.get_shape().size))

        normalizel2 = create_op_with_const_inputs(graph, NormalizeL2Op, {1: axes}, {'eps_mode': 'add', 'eps': node.eps})

        node.out_port(0).get_connection().set_source(mul.out_port(0))
        node.in_port(1).get_connection().get_source().connect(mul.in_port(1))
        normalizel2.out_port(0).connect(mul.in_port(0))
        node.in_port(0).get_connection().set_destination(normalizel2.in_port(0))

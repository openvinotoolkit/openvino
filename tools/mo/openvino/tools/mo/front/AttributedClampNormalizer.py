# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import float32_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_node
from openvino.tools.mo.ops.clamp import Clamp


class AttributedClampNormalizer(FrontReplacementPattern):
    """
    This transformation converts AttributedClamp operation (min/max are specified as attribute) to Clamp
    operation.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for attr_clamp in graph.get_op_nodes(op='AttributedClamp'):
            original_name = attr_clamp.soft_get('name', attr_clamp.id)

            rename_node(attr_clamp, original_name + '/TBR')
            min_value = attr_clamp.soft_get('min', np.finfo(np.float32).min)
            max_value = attr_clamp.soft_get('max', np.finfo(np.float32).max)
            new_clamp = create_op_with_const_inputs(graph, Clamp,
                                                    {1: float32_array(min_value),
                                                     2: float32_array(max_value)},
                                                    {'name': original_name})
            rename_node(new_clamp, original_name)

            attr_clamp.in_port(0).get_connection().set_destination(new_clamp.in_port(0))
            attr_clamp.out_port(0).get_connection().set_source(new_clamp.out_port(0))
            graph.remove_node(attr_clamp.id)

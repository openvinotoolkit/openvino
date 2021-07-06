# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from extensions.ops.elementwise import Sub, Div, Less, Round, Mul, Add, Greater
from extensions.ops.fakequantize import FakeQuantize
from extensions.ops.select import Select
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, Node
from mo.ops.const import Const


class FakeQuantWithMinMaxVarsToQuantize(FrontReplacementOp):
    """
    Performs FakeQuantize limits adjustment for min <= max following rules:
    If 0 < min < max: min_adj = 0 and max_adj = max - min.
    If min < max < 0: min_adj = min - max and max_adj = 0.
    If min <= 0 <= max:
        scale = (max - min) / (2^num_bits - 1),
        min_adj = scale * round(min / scale) and max_adj = max + min_adj - min.
    """
    op = "FakeQuantWithMinMaxVars"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: Dict[str, Node]):
        node = match['op']
        name = node.name

        min_port_tuple = (node.in_port(1).get_source().node, node.in_port(1).get_source().idx)
        max_port_tuple = (node.in_port(2).get_source().node, node.in_port(2).get_source().idx)

        node.in_port(1).disconnect()
        node.in_port(2).disconnect()

        # make sure min < max
        min_less_max = Less(graph, {'name': name + '/if_min_less_max'}).create_node([min_port_tuple, max_port_tuple])
        minimum = Select(graph, {'name': name + '/minimum'}).create_node([min_less_max, min_port_tuple, max_port_tuple])
        maximum = Select(graph, {'name': name + '/maximum'}).create_node([min_less_max, max_port_tuple, min_port_tuple])

        # to create zero of limits data type, we multiply it by integer zero
        zero = create_op_node_with_second_input(graph, Mul, int64_array(0), {'name': name + '/zero'}, input_node=minimum)

        # if 0 < min < max: min_adj = 0 and max_adj = max - min
        min_greater_zero = Greater(graph, {'name': name + '/if_minimum_greater_zero'}).create_node([minimum, zero])
        max_minus_min = Sub(graph, {'name': name + '/max_minus_min'}).create_node([maximum, minimum])
        minimum = Select(graph, {'name': name + '/first_adj_min'}).create_node([min_greater_zero, zero, minimum])
        maximum = Select(graph, {'name': name + '/first_adj_max'}).create_node([min_greater_zero, max_minus_min, maximum])

        # if min < max < 0: min_adj = min - max and max_adj = 0
        max_less_zero = Less(graph, {'name': name + '/if_max_less_zero'}).create_node([maximum, zero])
        min_minus_max = Sub(graph, {'name': name + '/min_minus_max'}).create_node([minimum, maximum])
        minimum = Select(graph, {'name': name + '/second_adj_min'}).create_node([max_less_zero, min_minus_max, minimum])
        maximum = Select(graph, {'name': name + '/second_adj_max'}).create_node([max_less_zero, zero, maximum])

        # scale = (max - min) / (2 ^ num_bits - 1),
        float_range = Sub(graph, {'name': name + '/float_range'}).create_node([maximum, minimum])
        quant_min_value, quant_max_value = int(node.narrow_range), 2 ** node.num_bits - 1
        int_range = Const(graph, dict(name=name + '/int_range', value=quant_max_value - quant_min_value)).create_node()
        scale = Div(graph, {'name': name + '/scale'}).create_node([float_range, int_range])
        # min_adj = scale * round(min / scale)
        descaled_min = Div(graph, {'name': name + '/descaled_min'}).create_node([minimum, scale])
        rounded_descaled_min = Round(graph, {'name': name + '/rounded_descaled_min'}).create_node([descaled_min])
        min_adj = Mul(graph, {'name': name + '/min_adj'}).create_node([scale, rounded_descaled_min])
        # max_adj = max + min_adj - min.
        adjustment = Sub(graph, {'name': name + '/limits_adjustment'}).create_node([min_adj, minimum])
        max_adj = Add(graph, {'name': name + '/max_adj'}).create_node([maximum, adjustment])

        # FakeQuantize operation has 5 inputs instead of 3 inputs in TensorFlow
        node.add_input_port(3, skip_if_exist=True)
        node.add_input_port(4, skip_if_exist=True)

        node.in_port(1).connect(min_adj.out_port(0))
        node.in_port(2).connect(max_adj.out_port(0))
        node.in_port(3).connect(min_adj.out_port(0))
        node.in_port(4).connect(max_adj.out_port(0))

        FakeQuantize.update_node_stat(node, {'levels': node['levels']})

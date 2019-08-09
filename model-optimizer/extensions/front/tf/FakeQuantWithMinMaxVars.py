"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Dict

from extensions.ops.elementwise import Sub, Div, Less, Greater, Round, Mul
from extensions.ops.fakequantize import FakeQuantize
from extensions.ops.select import Select
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.const import Const


class FakeQuantWithMinMaxVarsToQuantize(FrontReplacementOp):
    op = "FakeQuantWithMinMaxVars"
    enabled = True

    def run_before(self):
        from extensions.front.sub import Sub
        from extensions.front.div import Div
        return [Sub, Div]

    def replace_sub_graph(self, graph: Graph, match: Dict[str, Node]):
        node = match['op']
        name = node.name

        # Zero Point Nudging : Scale counting
        f_min = node.in_port(1).get_source()
        node.in_port(1).disconnect()
        f_max = node.in_port(2).get_source()
        node.in_port(2).disconnect()

        f_diff = Sub(graph, {'name': name + '/float_range'}).create_node()
        f_max.connect(f_diff.in_port(0))
        f_min.connect(f_diff.in_port(1))

        quant_min_value = int(node.narrow_range)
        quant_max_value = 2 ** node.num_bits - 1
        i_diff = Const(graph, dict(name=name + '/int_range', value=quant_max_value - quant_min_value)).create_node()

        scale = Div(graph, {'name': name + '/scale'}).create_node()
        f_diff.out_port(0).connect(scale.in_port(0))
        i_diff.out_port(0).connect(scale.in_port(1))

        # Zero Point Nudging : ZP from min counting
        descaled_min = Div(graph, {'name': name + '/descaled_min'}).create_node()
        f_min.connect(descaled_min.in_port(0))
        scale.out_port(0).connect(descaled_min.in_port(1))

        zero_point_from_min = Sub(graph, {'name': name + '/zero_point_from_min'}).create_node()
        quant_min = Const(graph, {'value': quant_min_value, 'name': name + '/quant_min'}).create_node()
        quant_min.out_port(0).connect(zero_point_from_min.in_port(0))
        descaled_min.out_port(0).connect(zero_point_from_min.in_port(1))

        # Zero Point Nudging : Nudged Zero Point counting
        zp_lesser_q_mi = Less(graph, {'name': name + '/zero_point_from_min_less_quant_min'}).create_node()
        zero_point_from_min.out_port(0).connect(zp_lesser_q_mi.in_port(0))
        quant_min.out_port(0).connect(zp_lesser_q_mi.in_port(1))

        zp_greater_q_ma = Greater(graph, {'name': name + '/zero_point_from_min_greater_quant_max'}).create_node()
        zero_point_from_min.out_port(0).connect(zp_greater_q_ma.in_port(0))
        quant_max = Const(graph, {'value': quant_max_value, 'name': name + '/quant_max'}).create_node()
        quant_max.out_port(0).connect(zp_greater_q_ma.in_port(1))

        rounded_zero_point_from_min = Round(graph, {'name': name + '/zero_point_from_min_rounding'}).create_node()
        zero_point_from_min.out_port(0).connect(rounded_zero_point_from_min.in_port(0))

        nudged_zero_point = Select(graph, {'name': name + '/nudging_zp_1_select_less_condition'}).create_node()
        greater_condition = Select(graph, {'name': name + '/nudging_zp_2_select_greater_condition'}).create_node()

        greater_condition.in_port(0).connect(zp_greater_q_ma.out_port(0))
        greater_condition.in_port(1).connect(quant_max.out_port(0))
        greater_condition.in_port(2).connect(rounded_zero_point_from_min.out_port(0))

        nudged_zero_point.in_port(0).connect(zp_lesser_q_mi.out_port(0))
        nudged_zero_point.in_port(1).connect(quant_max.out_port(0))
        nudged_zero_point.in_port(2).connect(greater_condition.out_port(0))

        nudged_i_min = Sub(graph, {'name': name + '/nudged_i_min'}).create_node()
        quant_min.out_port(0).connect(nudged_i_min.in_port(0))
        nudged_zero_point.out_port(0).connect(nudged_i_min.in_port(1))

        nudged_i_max = Sub(graph, {'name': name + '/nudged_i_max'}).create_node()
        quant_max.out_port(0).connect(nudged_i_max.in_port(0))
        nudged_zero_point.out_port(0).connect(nudged_i_max.in_port(1))

        nudged_min = Mul(graph, {'name': name + '/nudged_min'}).create_node()
        nudged_i_min.out_port(0).connect(nudged_min.in_port(0))
        scale.out_port(0).connect(nudged_min.in_port(1))

        nudged_max = Mul(graph, {'name': name + '/nudged_max'}).create_node()
        nudged_i_max.out_port(0).connect(nudged_max.in_port(0))
        scale.out_port(0).connect(nudged_max.in_port(1))

        nudged_min.out_port(0).connect(node.in_port(1))
        nudged_max.out_port(0).connect(node.in_port(2))

        # FakeQuantize operation has 5 inputs instead of 3 inputs in TensorFlow
        node.add_input_port(3, skip_if_exist=True)
        node.add_input_port(4, skip_if_exist=True)

        node.in_port(3).connect(nudged_min.out_port(0))
        node.in_port(4).connect(nudged_max.out_port(0))

        FakeQuantize.update_node_stat(node, {'levels': node['levels']})

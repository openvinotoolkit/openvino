"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.ops.elementwise import Div, Add, Sub, Mul
from mo.front.common.partial_infer.utils import float_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Node, Graph
from mo.ops.assign import Assign
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.read_value import ReadValue
from mo.ops.result import Result
import numpy as np

class StatsExtractPoolReplacer(FrontReplacementSubgraph):
    enabled = True

    def run_after(self):
        from extensions.front.kaldi.memory_offset_adjustment import MemoryOffsetAdjustment

        return [MemoryOffsetAdjustment]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='statisticsextractioncomponent'):
            self.replace_stats(graph, node)

    def replace_stats(self, graph: Graph, stats_node: Node):
        stats_node_name = stats_node.soft_get('name', stats_node.id)
        assert stats_node.has_and_set('input_dim'), 'statistics extracting does not have set input_dim'
        default_zeros = np.zeros([1, stats_node.input_dim], dtype=np.float32)

        # count infer requests
        count_name = stats_node_name + '/count/'
        const_one = Const(graph, {'name': count_name + 'one', 'value': float_array([1])}).create_node()
        default_count = Const(graph, {'name': count_name + 'default_count', 'value': float_array([1])}).create_node()
        count = ReadValue(graph, {'name': count_name + 'read(count)', 'variable_id': count_name}).create_node()
        count_assign = Assign(graph, {'name': count_name + 'assign(count)', 'variable_id': count_name}).create_node()
        res_node = Result(graph, {'name': count_name + 'assign(count)/fake_output'}).create_node()
        count_assign.out_port(0).connect(res_node.in_port(0))  # so that Assign will not be deleted

        add = Add(graph, {'name': count_name + 'Add'}).create_node()  # for num_iter

        default_count.out_port(0).connect(count.in_port(0))
        count.out_port(0).connect(add.in_port(0))
        const_one.out_port(0).connect(add.in_port(1))
        add.out_port(0).connect(count_assign.in_port(0))

        # graph for Welford's  algorithm for online mean and variance calculation
        # Mn - accumulated mean from the last infer request
        # Sn - accumulated var*count from the last infer request (Var = Sn/count)
        alg_name = stats_node_name + '/welford_algo/'
        Mn = ReadValue(graph, {'name': alg_name + 'read(Mn)', 'variable_id': alg_name + 'Mn'}).create_node()
        assign_new_Mn = Assign(graph, {'name': alg_name + 'assign(M(n+1))', 'variable_id': alg_name + 'Mn'}).create_node()
        res_node = Result(graph, {'name': alg_name + 'assign(M(n+1))/fake_output'}).create_node()
        assign_new_Mn.out_port(0).connect(res_node.in_port(0))  # so that Assign will not be deleted

        Sn = ReadValue(graph, {'name': alg_name + 'read(Sn)', 'variable_id': alg_name + 'Sn'}).create_node()
        assign_new_Sn = Assign(graph, {'name': alg_name + 'assign(S(n+1))', 'variable_id': alg_name + 'Sn'}).create_node()
        res_node = Result(graph, {'name': alg_name + 'assign(S(n+1))/fake_output'}).create_node()
        assign_new_Sn.out_port(0).connect(res_node.in_port(0))  # so that Assign will not be deleted

        # default values of Mn and Sn are returned on the first infer request
        default_Mn = Const(graph, {'name': alg_name + 'default_Mn', 'value': default_zeros}).create_node()
        default_Sn = Const(graph, {'name': alg_name + 'default_Sn', 'value': default_zeros}).create_node()
        default_Mn.out_port(0).connect(Mn.in_port(0))
        default_Sn.out_port(0).connect(Sn.in_port(0))

        delta_1 = Sub(graph, {'name': alg_name + '/M(n-1)-x/'}).create_node()  # delta_1 = x - Mn
        delta_2 = Sub(graph, {'name': alg_name + '/Mn-x/'}).create_node()  # delta_2 = x - M(n)

        delta_1_div_count = Div(graph, {'name': alg_name + '/delta_1_div_count/'}).create_node()  # delta_1 / count
        var_n = Div(graph, {'name': alg_name + '/Sn_div_count/'}).create_node()  # Variance(n) = S(n) / count

        Mn_new = Add(graph, {'name': alg_name + '/Mn_new/'}).create_node()  # M(n+1) = Mn + delta_1 / count
        Sn_new = Add(graph, {'name': alg_name + '/Sn_new/'}).create_node()  # S(n+1) = Sn + delta_1 * delta_2
        delta_1_mul_delta_2 = Mul(graph, {'name': alg_name + '/delta1_mul_delta2/'}).create_node()  # delta_1*delta_2
        concat = Concat(graph, {'name': alg_name + '/concat_Mn_Sn/'}).create_node()
        concat.add_sequence_of_ports('in', range(2))

        # connect graph for Welford's algorithm
        delta_1.in_port(0).connect(stats_node.in_port(0).get_source())
        delta_1.in_port(1).connect(Mn.out_port(0))
        delta_2.in_port(0).connect(stats_node.in_port(0).get_source())
        delta_1.out_port(0).connect(delta_1_div_count.in_port(0))
        count.out_port(0).connect(delta_1_div_count.in_port(1))

        Mn.out_port(0).connect(Mn_new.in_port(0))
        delta_1_div_count.out_port(0).connect(Mn_new.in_port(1))
        Mn_new.out_port(0).connect(assign_new_Mn.in_port(0))
        Mn_new.out_port(0).connect(delta_2.in_port(1))  # delta_2 = x - M(n)

        delta_1.out_port(0).connect(delta_1_mul_delta_2.in_port(0))
        delta_2.out_port(0).connect(delta_1_mul_delta_2.in_port(1))
        Sn.out_port(0).connect(Sn_new.in_port(0))
        delta_1_mul_delta_2.out_port(0).connect(Sn_new.in_port(1))
        Sn_new.out_port(0).connect(assign_new_Sn.in_port(0))

        Sn_new.out_port(0).connect(var_n.in_port(0))  # Var = S(n+1) / count
        count.out_port(0).connect(var_n.in_port(1))  # Var = S(n+1) / count
        Mn_new.out_port(0).connect(concat.in_port(0))
        var_n.out_port(0).connect(concat.in_port(1))

        stats_node.out_node().out_port(0).get_connection().set_source(concat.out_port(0))
        stats_node.in_port(0).disconnect()

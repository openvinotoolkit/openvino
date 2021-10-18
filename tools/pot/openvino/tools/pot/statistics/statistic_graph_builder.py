# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.ops.result import Result
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.front.common.partial_infer.utils import int64_array
from extensions.ops.ReduceOps import ReduceMin, ReduceMax, ReduceMean
from extensions.ops.activation_ops import Abs

from ..graph.model_utils import get_node_by_name
from ..graph.node_utils import get_output_shape
from ..statistics.statistics import Statistic, TensorStatistic, TensorStatisticAxis
from ..statistics.function_selector import ACTIVATIONS, get_stats_function

from ..utils.utils import convert_output_key



class StatisticGraphBuilder:
    def insert_statistic(self, model, stats_layout, stat_aliases=None):
        if stat_aliases is None or model is None:
            return model, list(stats_layout.keys())
        nodes_names = []
        for algo_name, node_stats in stat_aliases.items():
            for node_name, stats in node_stats.items():
                node_name_in_graph = node_name[0] if isinstance(node_name, tuple) else node_name
                node = get_node_by_name(model, node_name_in_graph)
                node_name = convert_output_key(node_name)
                for stat, _ in list(stats.items()):
                    if not isinstance(stat, Statistic) or not stat.kwargs.get('inplace_statistics', False):
                        nodes_names.append(node_name_in_graph)
                        continue
                    type_stat = stat.kwargs['type']
                    add_output_node = getattr(self, f'insert_{type_stat}')(node, type_stat, node_name, **stat.kwargs)
                    if add_output_node:
                        nodes_names.append(add_output_node)
                        class_statistic = TensorStatistic if isinstance(stat, TensorStatistic) else TensorStatisticAxis

                        del stats_layout[node_name][stat]
                        fn = get_stats_function(ACTIVATIONS, type_stat, stat.kwargs.get('granularity'),
                                                'compute_statistic')
                        new_stat = class_statistic(fn,
                                                   channel=stat.kwargs.get('channel', {}),
                                                   inplace_statistics=False,
                                                   granularity=stat.kwargs.get('granularity'),
                                                   type=type_stat)
                        stats_layout[node_name][new_stat] = new_stat

                        stat_name = stat_aliases[algo_name][node_name][stat]
                        del stat_aliases[algo_name][node_name][stat]
                        stat_aliases[algo_name][node_name][new_stat] = stat_name

        return model, nodes_names

    def insert_reduce(self, insert_op, node, granularity, type_stat, node_name, axis=1):
        axis_const = self.find_axis(node, granularity, axis)
        if isinstance(axis_const, str):
            return node.name
        reduce_op = create_op_node_with_second_input(node.graph, insert_op, int64_array(axis_const),
                                                     dict(name=f'{type_stat}_{node_name}'))

        node.out_port(0).connect(reduce_op.in_port(0))
        self.insert_result(node, reduce_op, type_stat)
        return None

    def insert_min(self, node, type_stat, node_name, **kwargs):
        return self.insert_reduce(ReduceMin, node, kwargs.get('granularity'), type_stat, node_name)

    def insert_max(self, node, type_stat, node_name, **kwargs):
        return self.insert_reduce(ReduceMax, node, kwargs.get('granularity'), type_stat, node_name)

    def insert_mean(self, node, type_stat, node_name, **kwargs):
        axis_channel = kwargs.get('channel', None).get(node.name, 1)
        return self.insert_reduce(ReduceMean, node, kwargs.get('granularity'), type_stat, node_name, axis_channel)

    def insert_abs_max(self, node, type_stat, node_name, **kwargs):
        axis_const = self.find_axis(node, kwargs.get('granularity'))
        if isinstance(axis_const, str):
            return node.name
        abs_node = Abs(node.graph, {"name": type_stat + node_name}).create_node()
        abs_node.in_port(0).connect(node.out_port(0))
        max_op = create_op_node_with_second_input(node.graph, ReduceMax, int64_array(axis_const),
                                                  dict(name='abs_max_' + node.name))
        abs_node.out_port(0).connect(max_op.in_port(0))
        self.insert_result(node, max_op, type_stat)
        return None

    @staticmethod
    def insert_result(node, child_node, name):
        res_op = Result(node.graph, {'name': f'Result_{name}_{node.name}'}).create_node()
        child_node.out_port(0).connect(res_op.in_port(0))

    @staticmethod
    def find_axis(node, granularity, axis=1):
        shape = len(get_output_shape(node, 0))
        if shape < 3 and granularity == 'perchannel':
            return node.name
        axis_const = list(i for i in range(shape))
        if granularity == 'perchannel':
            axis_const.pop(axis)
            axis_const.pop(0)
        return axis_const

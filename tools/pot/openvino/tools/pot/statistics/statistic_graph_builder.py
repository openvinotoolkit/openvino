# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from mo.ops.result import Result
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.front.common.partial_infer.utils import int64_array
from extensions.back.add_outputs_recursive import AddOutputRecursive
from extensions.ops.ReduceOps import ReduceMin, ReduceMax, ReduceMean
from extensions.ops.activation_ops import Abs

from ..graph.editor import get_node_by_name_recursively
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
        output_to_node_names = {}
        copy_stat_aliases = deepcopy(stat_aliases)
        for algo_name, node_stats in copy_stat_aliases.items():
            for node_name, stats in node_stats.items():
                node_name_in_graph = node_name[0] if isinstance(node_name, tuple) else node_name
                pos = node_name_in_graph.find('/pre_fq_input')
                if pos != -1:
                    node_name_in_graph = node_name_in_graph[:pos]
                node = get_node_by_name(model, node_name_in_graph)
                node_name = convert_output_key(node_name)
                node_in_main_graph =  get_node_by_name(model, node_name_in_graph.split('|')[0])
                model_graph = node_in_main_graph.graph
                for stat, _ in list(stats.items()):
                    if not isinstance(stat, Statistic) or not stat.kwargs.get('inplace_statistics', False):
                        if node_name_in_graph not in nodes_names:
                            nodes_names.append(node_name_in_graph)
                        continue
                    type_stat = stat.kwargs['type']
                    # store original name instead of fullname
                    node_name_for_stats = (node.name, node_name[1]) if isinstance(node_name, tuple) else node.name
                    node_name_for_stats = convert_output_key(node_name_for_stats)

                    add_output_node = getattr(self, f'insert_{type_stat}')(model_graph,
                                                                           node,
                                                                           type_stat,
                                                                           node_name_for_stats,
                                                                           **stat.kwargs)
                    if add_output_node:
                        if node_name_in_graph not in nodes_names:
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

                # add output if node in subgraph
                if node_name_in_graph != node.name:
                    if node_name_in_graph in nodes_names:
                        nodes_names.remove(node_name_in_graph)
                    for model_dict in model.models:
                        if model_graph == model_dict['model']:
                            if model.is_cascade and node_name_in_graph.startswith(model_dict['name']):
                                original_fullname = node_name_in_graph.replace(model_dict['name'] + '_', '', 1)
                            else:
                                original_fullname = node_name_in_graph
                            model_graph.graph['additional_outputs'] = original_fullname.split('|')
                            results = AddOutputRecursive().find_and_replace_pattern(model_graph)
                            assert len(results) == 1
                            result_name = results[0].name.split(':')[0]
                            if node_name in stats_layout:
                                stats_layout[result_name] = stats_layout.pop(node_name)
                            stat_aliases[algo_name][result_name] = stat_aliases[algo_name].pop(node_name)
                            output_to_node_names[result_name] = node_name_in_graph
                            break

        return model, nodes_names, output_to_node_names

    def add_child_fullname(self, parent_node, child_node):
        child_node['fullname'] = '|'.join(parent_node.fullname.split('|')[:-1] + [child_node.name])

    def insert_reduce(self, main_graph, insert_op, node, granularity, type_stat, node_name, axis=1):
        axis_const = self.find_axis(node, granularity, axis)
        if isinstance(axis_const, str):
            return node.name
        reduce_op = create_op_node_with_second_input(node.graph, insert_op, int64_array(axis_const),
                                                     dict(name=f'{type_stat}_{node_name}'))
        self.add_child_fullname(node, reduce_op)
        node.out_port(0).connect(reduce_op.in_port(0))
        self.insert_result(main_graph, node, reduce_op, type_stat)
        return None

    def insert_min(self, main_graph, node, type_stat, node_name, **kwargs):
        return self.insert_reduce(main_graph, ReduceMin, node, kwargs.get('granularity'), type_stat, node_name)

    def insert_max(self, main_graph, node, type_stat, node_name, **kwargs):
        return self.insert_reduce(main_graph, ReduceMax, node, kwargs.get('granularity'), type_stat, node_name)

    def insert_mean(self, main_graph, node, type_stat, node_name, **kwargs):
        axis_channel = kwargs.get('channel', None).get(node.name, 1)
        return self.insert_reduce(main_graph, ReduceMean, node, kwargs.get('granularity'), type_stat, node_name, axis_channel)

    def insert_abs_max(self, main_graph, node, type_stat, node_name, **kwargs):
        axis_const = self.find_axis(node, kwargs.get('granularity'))
        if isinstance(axis_const, str):
            return node.name
        abs_node = Abs(node.graph, {"name": f'{type_stat}_{node_name}'}).create_node()
        abs_node.in_port(0).connect(node.out_port(0))
        max_op = create_op_node_with_second_input(node.graph, ReduceMax, int64_array(axis_const),
                                                  dict(name=f'abs_max_{node.name}'))
        abs_node.out_port(0).connect(max_op.in_port(0))
        self.insert_result(node, max_op, type_stat)
        self.insert_result(main_graph, node, max_op, type_stat)
        return None

    @staticmethod
    def insert_result(main_graph, node, child_node, name):
        if node.graph != main_graph:
            main_graph.graph['additional_outputs'] = child_node.fullname.split('|')
            AddOutputRecursive().find_and_replace_pattern(main_graph)
        else:
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

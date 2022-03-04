# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import re
import numpy as np

from ..statistics.statistics import compute_statistic, Statistic, TensorStatistic
from ..statistics.function_selector import get_stats_function, ACTIVATIONS
from ..utils.logger import get_logger
from ..utils.utils import convert_output_key

logger = get_logger(__name__)


def append_stats(accumulated_layer_stats, stats_layout, value, dataset_index, inference_for_shape):
    inplace_stats_mapping = get_inplace_stats_mapping(stats_layout)
    if isinstance(value, list):
        value = parse_sequential_stats(value, stats_layout, inference_for_shape)
    else:
        value = process_raw_output(value)
    for layer, stats in stats_layout.items():
        if layer not in accumulated_layer_stats:
            accumulated_layer_stats[layer] = {stat_name: [] for stat_name in stats_layout[layer]}
        for stat_name, stat_fn in stats.items():
            layer_stat_name = inplace_stats_mapping[layer][stat_name]
            if layer_stat_name in value:
                accumulated_layer_stats[layer][stat_name].append(
                    (dataset_index, compute_statistic(stat_fn, value, layer_stat_name)))


def parse_sequential_stats(value_sequential, stats_layout, inference_for_shape):
    stat_names_by_layer, old_names_mapping = get_per_layer_stat_mapping(stats_layout)
    activation_seq = defaultdict(lambda: [])
    for value in value_sequential:
        value = process_raw_output(value)
        for layer, activations in value.items():
            get_sequential_activations(activations, layer, activation_seq, stats_layout,
                                       stat_names_by_layer, old_names_mapping)

    for layer, act_seq in activation_seq.items():
        seq_len = len(act_seq[0].shape)
        if inference_for_shape:
            activation_seq[layer] = act_seq[0]
            continue
        if not isinstance(stat_names_by_layer[layer], Statistic) or \
                not stat_names_by_layer[layer].kwargs.get('inplace_statistics', False):
            axis = 1 if seq_len == 2 else 2
        else:
            axis = seq_len if seq_len in (0, 1) else 2
        activation_seq[layer] = np.stack(act_seq, axis=axis)
    return activation_seq


def process_accumulated_stats(accumulated_stats, stat_names_aliases=None):
    for layer in accumulated_stats:
        for stat in accumulated_stats[layer]:
            accumulated_stats[layer][stat].sort(key=lambda el: el[0])
            accumulated_stats[layer][stat] = [el[1] for el in accumulated_stats[layer][stat]]

    # pack IE-like stats names into original ones
    if stat_names_aliases is not None:
        accumulated_stats = {stat_names_aliases[key]: value
                             for key, value in accumulated_stats.items()}

    return accumulated_stats


def get_per_layer_stat_mapping(stats_layout):
    old_names_mapping, stat_names_by_layer = {}, {}
    for layer, stats in stats_layout.items():
        for stat_name, _ in stats.items():
            layer_stat_name = layer
            if hasattr(stat_name, 'kwargs') and stat_name.kwargs.get('inplace_statistics', False):
                layer_stat_name = stat_name.kwargs.get('layer_stat_name', stat_name.kwargs['type'] + '_' + layer)
            old_names_mapping[layer_stat_name], stat_names_by_layer[layer_stat_name] = layer, stat_name

    return stat_names_by_layer, old_names_mapping


def get_inplace_stats_mapping(stats_layout):
    old_name_stat_new_name = {}
    for layer, stats in stats_layout.items():
        for stat_name, _ in stats.items():
            layer_stat_name = layer
            if hasattr(stat_name, 'kwargs') and stat_name.kwargs.get('inplace_statistics', False):
                layer_stat_name = stat_name.kwargs.get('layer_stat_name', stat_name.kwargs['type'] + '_' + layer)
            if layer not in old_name_stat_new_name:
                old_name_stat_new_name[layer] = {stat_name: layer_stat_name}
            else:
                old_name_stat_new_name[layer][stat_name] = layer_stat_name

    return old_name_stat_new_name


def get_sequential_activations(activations, layer, activation_seq, stats_layout,
                               stat_names_by_layer, old_names_mapping):
    if old_names_mapping.get(layer, None) in stats_layout and hasattr(stat_names_by_layer[layer], 'kwargs') \
            and stat_names_by_layer[layer].kwargs.get('inplace_statistics', False):
        granularity, fn_type = stat_names_by_layer[layer].kwargs.get('granularity', None), stat_names_by_layer[
            layer].kwargs.get('type', None)
        fn = get_stats_function(ACTIVATIONS, fn_type, granularity, 'compute_statistic')
        stats_layout[old_names_mapping[layer]].pop(stat_names_by_layer[layer])
        stats_layout[old_names_mapping[layer]][stat_names_by_layer[layer]] = TensorStatistic(fn)
        activation_seq[layer].append(activations)
    elif old_names_mapping.get(layer, None) in stats_layout and hasattr(stat_names_by_layer[layer], 'kwargs') \
            and not stat_names_by_layer[layer].kwargs.get('inplace_statistics', False):
        activation_seq[layer].append(activations)
    elif old_names_mapping.get(layer, None) in stats_layout and (callable(stat_names_by_layer[layer]) \
            or callable(stats_layout[layer][stat_names_by_layer[layer]])):
        activation_seq[layer].append(activations)


def update_stats(stats_layout: dict, stat_aliases: dict, old_key: str, new_key: str):
    stats_layout[new_key] = stats_layout.pop(old_key)
    for algo_name in stat_aliases:
        if old_key in stat_aliases[algo_name]:
            stat_aliases[algo_name][new_key] = stat_aliases[algo_name].pop(old_key)


def restore_original_node_names(output2node, accumulated_stats, stats_layout, stat_aliases):
    if output2node and stats_layout:
        for out_name, original_node_name in output2node.items():
            accumulated_stats[original_node_name] = accumulated_stats.pop(out_name)
            update_stats(stats_layout, stat_aliases, out_name, original_node_name)


def align_stat_names_with_results(result_names, nodes_name, output2node, stats_layout, stat_aliases):
    """ Change node name in stast to result name if in the original model the subgraph had 1 output,
    but after adding outputs in the subgraph, the number of output ports increased.
    For such nodes, it is necessary to add a '.0' to the original output name
    :param: result_names: names of Result nodes
    :param: nodes_name: node name in graph
    :param: output2node: a dict storing the matching of the result to the node
    :param: stats_layout: dict of stats collection functions
    :param: stat_aliases: dict of algorithms collections stats
    """
    if output2node:
        for original_out_name in nodes_name:
            if original_out_name not in result_names and (original_out_name, 0) not in stats_layout:
                out_name_with_port = original_out_name + '.0'
                assert out_name_with_port in result_names
                update_stats(stats_layout, stat_aliases, original_out_name, out_name_with_port)
                output2node[out_name_with_port] = original_out_name


def process_raw_output(raw_output):
    """ Process raw output into the POT friendly format """
    result = {}
    for result_node, result_data in raw_output.items():
        for name in result_node.get_tensor().get_names():
            result_name = get_clean_name(name)
            result[result_name] = result_data
    return result


def add_tensor_names(nodes, names):
    """ Process nGraph nodes and sets POT-friendly tensor name """
    for ng_node, node_name in zip(nodes, names):
        names = ng_node.get_tensor().get_names()
        names.add(convert_output_key(node_name))
        ng_node.get_tensor().set_names(names)

def cast_friendly_names(nodes):
    """ Process nGraph nodes and sets POT-friendly tensor name
    based on friendly_name
     """
    for ng_node in nodes:
        names = ng_node.get_tensor().get_names()
        names.add(ng_node.get_node().friendly_name)
        ng_node.get_tensor().set_names(names)

def collect_model_outputs(ng_model):
    """ Collect nGraph model outputs and their tensor names
    """
    model_output_names = []
    for ng_output in ng_model.outputs:
        model_output_names.extend(list(ng_output.get_tensor().get_names()))
    return model_output_names

def get_clean_name(name):
    return re.sub(r'/sink_port_\d+', '', name)

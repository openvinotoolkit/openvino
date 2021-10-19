# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import numpy as np

from ..statistics.statistics import compute_statistic, Statistic, TensorStatistic
from ..statistics.function_selector import get_stats_function, ACTIVATIONS
from ..utils.logger import get_logger

logger = get_logger(__name__)


def append_stats(accumulated_layer_stats, stats_layout, value, dataset_index):
    inplace_stats_mapping = get_inplace_stats_mapping(stats_layout)
    if isinstance(value, list):
        value = parse_sequential_stats(value, stats_layout)

    for layer, stats in stats_layout.items():
        if layer not in accumulated_layer_stats:
            accumulated_layer_stats[layer] = {stat_name: [] for stat_name in stats_layout[layer]}
        for stat_name, stat_fn in stats.items():
            layer_stat_name = inplace_stats_mapping[layer][stat_name]
            if layer_stat_name in value:
                accumulated_layer_stats[layer][stat_name].append(
                    (dataset_index, compute_statistic(stat_fn, value, layer_stat_name)))


def parse_sequential_stats(value_sequential, stats_layout):
    stat_names_by_layer, old_names_mapping = get_per_layer_stat_mapping(stats_layout)
    activation_seq = defaultdict(lambda: [])
    for value in value_sequential:
        for layer, activations in value.items():
            get_sequential_activations(activations, layer, activation_seq, stats_layout,
                                       stat_names_by_layer, old_names_mapping)

    for layer, act_seq in activation_seq.items():
        if not isinstance(stat_names_by_layer[layer], Statistic) or \
                not stat_names_by_layer[layer].kwargs.get('inplace_statistics', False):
            axis = 1 if len(act_seq[0].shape) == 2 else 2
        else:
            axis = 1 if len(act_seq[0].shape) == 1 else 2
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
                layer_stat_name = stat_name.kwargs['type'] + '_' + layer
            old_names_mapping[layer_stat_name], stat_names_by_layer[layer_stat_name] = layer, stat_name

    return stat_names_by_layer, old_names_mapping


def get_inplace_stats_mapping(stats_layout):
    old_name_stat_new_name = {}
    for layer, stats in stats_layout.items():
        for stat_name, _ in stats.items():
            layer_stat_name = layer
            if hasattr(stat_name, 'kwargs') and stat_name.kwargs.get('inplace_statistics', False):
                layer_stat_name = stat_name.kwargs['type'] + '_' + layer
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
    elif old_names_mapping.get(layer, None) in stats_layout and callable(stat_names_by_layer[layer]):
        activation_seq[layer].append(activations)

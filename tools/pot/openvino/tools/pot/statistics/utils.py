# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def merge_algos_by_samplers(samplers):
    """
    Helper to sort all algos by its samplers and subset indices
    :param samplers: dict with algo names as key and corresponding sampler as value
    :return: list of tuples. Every tuple contains with list of algos
     corresponding to sampler and sampler itself.
    """
    predict_iterations = []
    for algo_name, sampler in samplers.items():
        append_iteration = True
        for iteration in predict_iterations:
            algo_seq, iter_sampler = iteration
            if isinstance(sampler, type(iter_sampler)) and \
                    sampler.num_samples is not None and \
                    sampler.num_samples == iter_sampler.num_samples:
                # add algorithm into predict_iterations by reference
                algo_seq.append(algo_name)
                append_iteration = False

        if append_iteration:
            predict_iterations.append(([algo_name], sampler))

    return predict_iterations


def merge_stats_by_algo_names(algo_names, stats_layout_by_algo):
    """
    Helper function to collect all stat functions from stats layout
     by algo names and intersect them for more efficient inference
    :param algo_names: list of algo names
    :param stats_layout_by_algo: dict with algo names as keys and list of stat functions as values
    :return: 2 element tuple. First element is dict with intersected statistics and the second is a
     dict with mapping statistic names into statistic keys (functions actually)
    """
    merged_stats = {}
    stat_aliases = {name: {} for name in algo_names}

    def add_node_stats(algo_name, node_name, stats):
        stat_aliases[algo_name][node_name] = {}
        node_state_aliases = stat_aliases[algo_name][node_name]
        if node_name not in merged_stats:
            merged_stats[node_name] = {}
        for stat_name, stat_fn in stats.items():
            merged_stats[node_name][stat_fn] = stat_fn
            # make stat aliases
            if stat_fn not in node_state_aliases:
                node_state_aliases[stat_fn] = list()
            node_state_aliases[stat_fn].append(stat_name)

    for name in algo_names:
        stat_aliases[name] = {node_name: {} for node_name in stats_layout_by_algo[name]}
        for node_name_, stats_ in stats_layout_by_algo[name].items():
            add_node_stats(name, node_name_, stats_)
    return merged_stats, stat_aliases

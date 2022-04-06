# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import numpy as np

from ....graph import node_utils as nu


def get_propagated_input_fq(node):
    def get_node_parents(node):
        return [n for n in nu.get_node_inputs(node) if n is not None and n.type != 'Const']

    def walk_to_parents(node):
        node_parents = get_node_parents(node)
        if node.type == 'FakeQuantize' and nu.get_node_input(node, 0).type != 'Const':
            input_fqs.append(node)
            return
        for node_parent in node_parents:
            walk_to_parents(node_parent)

    input_fqs = []
    for input_node in get_node_parents(node):
        walk_to_parents(input_node)
    return input_fqs


def compute_scale(fq):
    output_low = nu.get_node_input(fq, 3)
    output_high = nu.get_node_input(fq, 4)
    output_low_value = nu.get_node_value(output_low)
    output_high_value = nu.get_node_value(output_high)
    return np.max((output_high_value - output_low_value) / (fq.levels - 1))


def correct_node_overflow(weighted_node, node_statistics):
    weight_fq = nu.get_node_input(weighted_node, 1)
    weights_node = nu.get_node_input(weight_fq, 0)
    weights = deepcopy(nu.get_node_value(weights_node))
    weights_dtype = weights.dtype
    w_output_low = nu.get_node_input(weight_fq, 3)
    w_output_high = nu.get_node_input(weight_fq, 4)
    w_output_low_value = deepcopy(nu.get_node_value(w_output_low))
    w_output_high_value = deepcopy(nu.get_node_value(w_output_high))
    weight_scale = compute_scale(weight_fq)

    # Get input FQ node and compute input scale (in_scale)
    input_fqs = get_propagated_input_fq(weighted_node)
    input_scales = []
    for input_fq in input_fqs:
        input_scales.append(compute_scale(input_fq))
    input_scale = np.max(input_scales)

    # Get maximum values
    after_add_output = np.max(node_statistics)
    max_quantized_output = int(after_add_output / (input_scale * weight_scale))
    int32_type_max = np.iinfo(np.int32).max

    rescale_value = None

    if max_quantized_output > int32_type_max:
        rescale_value = max_quantized_output / int32_type_max
        weights = weights / rescale_value

        w_output_low_value = w_output_low_value * rescale_value
        w_output_high_value = w_output_high_value * rescale_value
        nu.set_node_value(weights_node, np.array(weights, dtype=weights_dtype))
        nu.set_node_value(w_output_low, w_output_low_value)
        nu.set_node_value(w_output_high, w_output_high_value)

    return rescale_value

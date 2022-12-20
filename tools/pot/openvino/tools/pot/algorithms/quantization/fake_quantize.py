# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import copy, deepcopy

import numpy as np
from addict import Dict
from bisect import bisect_left

from .fake_quantize_configuration import read_all_fake_quantize_configurations, get_configurations_by_preset, \
    get_configurations_by_qscheme, find_fqs_to_unify, add_range_estimator_configs, change_configurations_by_model_type
from .utils import load_hardware_config, merge_nested_dicts, get_ignored_operations
from ...graph.model_utils import get_nodes_by_type, get_node_by_name
from ...graph.node_utils import get_node_input, set_node_value, \
    get_node_value, get_node_output, get_node_inputs, get_input_shape, \
    get_quantized_input_key, get_input_data_value, get_first_convolutions
from ...graph.special_operations import OPERATIONS_WITH_WEIGHTS, TRANSPOSED_OPERATIONS
from ...graph.transformer import GraphTransformer
from ...utils.logger import get_logger

logger = get_logger(__name__)


def tune_range(a, b, num_bits):
    """ Tunes asymmetric quantization range to set zero quant precisely to zero value.
    Function moves left or right borders to do this and doesn't make left border higher or
    right border lesser than its original values
    :param a: range left border
    :param b: range right border
    :param num_bits: number of bits to perform quantization
    :return tuple with recomputed ranges
    """
    level_high = 2 ** num_bits - 1
    s = level_high / (b - a)
    fval = -a * s
    qval = np.round(fval)

    with np.errstate(invalid='ignore', divide='ignore'):
        ra = np.where(qval < level_high, qval / (qval - level_high) * b, a)
        rb = np.where(qval > 0.0, (qval - level_high) / qval * a, b)

    range_a = b - ra
    range_b = rb - a

    mask = np.where(range_a > range_b, 1.0, 0.0)
    inv_mask = np.abs(1.0 - mask)

    ra = mask * ra + inv_mask * a
    rb = inv_mask * rb + mask * b

    return ra, rb


def tune_range_unify_zp(a, b, num_bits):
    """ Tunes asymmetric quantization range to unify the zero point of all channels.
    Function is used to solve the per-tensor zero point constrain on KMB (vpu2p0)
    Function calculates the average zero point of all channels and tune the max/min range.
    Function moves left or right borders to do this and doesn't make left border higher or
    right border lesser than its original values
    :param a: range left border
    :param b: range right border
    :param num_bits: number of bits to perform quantization
    :return tuple with recomputed ranges
    """
    level_high = 2 ** num_bits - 1

    scale = (b - a) / level_high
    zero_point = -a / scale
    avg_zpts = np.round(np.mean(zero_point))

    qval = np.ones_like(a) * avg_zpts

    with np.errstate(invalid='ignore', divide='ignore'):
        ra = np.where(qval < level_high, qval / (qval - level_high) * b, a)
        rb = np.where(qval > 0.0, (qval - level_high) / qval * a, b)

    range_a = b - ra
    range_b = rb - a

    mask = np.where(range_a > range_b, 1.0, 0.0)
    inv_mask = np.abs(1.0 - mask)

    ra = mask * ra + inv_mask * a
    rb = inv_mask * rb + mask * b
    return ra, rb


def find_closest(arr, num):
    pos = bisect_left(arr, num)
    if pos == 0:
        return arr[0]
    if pos == len(arr):
        return arr[-1]
    before = arr[pos - 1]
    after = arr[pos]
    if after - num < num - before:
        return after
    else:
        return before


def find_closest_quantize(quants, data):
    res = [0] * len(data)
    for i, val in enumerate(data):
        res[i] = find_closest(quants, val)
    return res


def compute_relative_error(target, predict):
    th = 0.00001
    max_val = max(np.abs(target))
    if max_val < th:
        max_val = th
    res = 0.0
    for i in range(len(target)):
        diff = abs(target[i] - predict[i])

        if abs(target[i]) > th:
            diff /= abs(target[i])
        else:
            diff /= th
        diff *= abs(target[i]) / max_val
        res += diff
    return res / max(1, len(target))


def compute_best_scale(data):
    quants = [-28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0,
              -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0,
              -1.875, -1.75, -1.625, -1.5, -1.375, -1.25, -1.125, -1.0, -0.9375, -0.875, -0.8125, -0.75, -0.6875,
              -0.625, -0.5625, -0.5, -0.46875, -0.4375, -0.40625, -0.375, -0.34375, -0.3125, -0.28125, -0.25, -0.234375,
              -0.21875, -0.203125, -0.1875, -0.171875, -0.15625, -0.140625, -0.125, -0.117188, -0.109375, -0.101562,
              -0.09375, -0.0859375, -0.078125, -0.0703125, -0.0625, -0.0585938, -0.0546875, -0.0507812, -0.046875,
              -0.0429688, -0.0390625, -0.0351562, -0.03125, -0.0292969, -0.0273438, -0.0253906, -0.0234375, -0.0214844,
              -0.0195312, -0.0175781, -0.015625, -0.0146484, -0.0136719, -0.0126953, -0.0117188, -0.0107422,
              -0.00976562, -0.00878906, -0.0078125, -0.00732422, -0.00683594, -0.00634766, -0.00585938, -0.00537109,
              -0.00488281, -0.00439453, -0.00390625, -0.00366211, -0.00341797, -0.00317383, -0.00292969, -0.00268555,
              -0.00244141, -0.00219727, -0.00195312, -0.00183105, -0.00170898, -0.00158691, -0.00146484, -0.00134277,
              -0.0012207, -0.00109863, -0.000976562, -0.000854492, -0.000732422, -0.000610352, -0.000488281,
              -0.000366211, -0.000244141, -0.00012207, 0.0, 0.00012207, 0.000244141, 0.000366211, 0.000488281,
              0.000610352, 0.000732422, 0.000854492, 0.000976562, 0.00109863, 0.0012207, 0.00134277, 0.00146484,
              0.00158691, 0.00170898, 0.00183105, 0.00195312, 0.00219727, 0.00244141, 0.00268555, 0.00292969,
              0.00317383, 0.00341797, 0.00366211, 0.00390625, 0.00439453, 0.00488281, 0.00537109, 0.00585938,
              0.00634766, 0.00683594, 0.00732422, 0.0078125, 0.00878906, 0.00976562, 0.0107422, 0.0117188, 0.0126953,
              0.0136719, 0.0146484, 0.015625, 0.0175781, 0.0195312, 0.0214844, 0.0234375, 0.0253906, 0.0273438,
              0.0292969, 0.03125, 0.0351562, 0.0390625, 0.0429688, 0.046875, 0.0507812, 0.0546875, 0.0585938, 0.0625,
              0.0703125, 0.078125, 0.0859375, 0.09375, 0.101562, 0.109375, 0.117188, 0.125, 0.140625, 0.15625, 0.171875,
              0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5,
              0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0,
              2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0,
              13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    max_scale = 2 ** 5
    min_scale = 2 ** (-5)

    scales = []
    scale = max_scale
    while scale >= min_scale:
        scales.append(scale)
        scale /= 2

    min_diff = 100
    best_scale = max_scale

    for scale in scales:
        scaled_data = scale * data[:]
        quantized_data = find_closest_quantize(quants, scaled_data)
        diff = compute_relative_error(scaled_data, quantized_data)
        if diff < min_diff:
            min_diff = diff
            best_scale = scale

    return best_scale


def compute_best_scales(data):
    out_channels = data.shape[0]
    res = []
    for i in range(out_channels):
        row = np.copy(data[i]).flatten()
        if np.max(np.abs(row)) > 8:
            best_scale = 1.0 / np.max(np.abs(row))  # compute_best_scale(row)
            res.append(best_scale)
        else:
            res.append(1.0)
    return np.array(res)


def compute_scale(node):
    weights = None
    if node.in_port(0).get_source() is not None:
        weights = node.in_port(0).get_source().node

    if weights is None:
        return None
    try:
        data = weights.value
    except:
        return None

    return compute_best_scales(data)


def round_scale(value):
    quants = [-28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0,
              -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0,
              -1.875, -1.75, -1.625, -1.5, -1.375, -1.25, -1.125, -1.0, -0.9375, -0.875, -0.8125, -0.75, -0.6875,
              -0.625, -0.5625, -0.5, -0.46875, -0.4375, -0.40625, -0.375, -0.34375, -0.3125, -0.28125, -0.25, -0.234375,
              -0.21875, -0.203125, -0.1875, -0.171875, -0.15625, -0.140625, -0.125, -0.117188, -0.109375, -0.101562,
              -0.09375, -0.0859375, -0.078125, -0.0703125, -0.0625, -0.0585938, -0.0546875, -0.0507812, -0.046875,
              -0.0429688, -0.0390625, -0.0351562, -0.03125, -0.0292969, -0.0273438, -0.0253906, -0.0234375, -0.0214844,
              -0.0195312, -0.0175781, -0.015625, -0.0146484, -0.0136719, -0.0126953, -0.0117188, -0.0107422,
              -0.00976562, -0.00878906, -0.0078125, -0.00732422, -0.00683594, -0.00634766, -0.00585938, -0.00537109,
              -0.00488281, -0.00439453, -0.00390625, -0.00366211, -0.00341797, -0.00317383, -0.00292969, -0.00268555,
              -0.00244141, -0.00219727, -0.00195312, -0.00183105, -0.00170898, -0.00158691, -0.00146484, -0.00134277,
              -0.0012207, -0.00109863, -0.000976562, -0.000854492, -0.000732422, -0.000610352, -0.000488281,
              -0.000366211, -0.000244141, -0.00012207, 0.0, 0.00012207, 0.000244141, 0.000366211, 0.000488281,
              0.000610352, 0.000732422, 0.000854492, 0.000976562, 0.00109863, 0.0012207, 0.00134277, 0.00146484,
              0.00158691, 0.00170898, 0.00183105, 0.00195312, 0.00219727, 0.00244141, 0.00268555, 0.00292969,
              0.00317383, 0.00341797, 0.00366211, 0.00390625, 0.00439453, 0.00488281, 0.00537109, 0.00585938,
              0.00634766, 0.00683594, 0.00732422, 0.0078125, 0.00878906, 0.00976562, 0.0107422, 0.0117188, 0.0126953,
              0.0136719, 0.0146484, 0.015625, 0.0175781, 0.0195312, 0.0214844, 0.0234375, 0.0253906, 0.0273438,
              0.0292969, 0.03125, 0.0351562, 0.0390625, 0.0429688, 0.046875, 0.0507812, 0.0546875, 0.0585938, 0.0625,
              0.0703125, 0.078125, 0.0859375, 0.09375, 0.101562, 0.109375, 0.117188, 0.125, 0.140625, 0.15625, 0.171875,
              0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5,
              0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0,
              2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0,
              13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]

    quantized_data = find_closest(quants, value)

    return quantized_data


def round_scales(scale):
    rnd_fn = np.vectorize(round_scale)
    return rnd_fn(scale)


def fill_fake_quantize_node(fq, min_level, max_level, output_low=None, output_high=None):
    """ Fills fake quantize input nodes with min/max values
    :param fq: fake quantize node to fill
    :param min_level: low border of quantization range
    :param max_level: high border of quantization range
    """
    min_level_mean = np.min(min_level)
    max_level_mean = np.max(max_level)

    max_vals = {"hf8_ext": 28, "hf8_libxsmm": 448, "bf8": 57344}
    fq.destination_type = 'hf8_ext'  # 'hf8_libxsmm' #'hf8_ext'

    scale = max_vals[fq.destination_type] / np.maximum(max_level, np.abs(min_level) + np.finfo(float).eps)
    fq.apply_scale = True

    print(fq.name, ' th value: ', fq.destination_type, fq.apply_scale, scale.shape)

    def _update_node_val(port_idx, value):
        _node = get_node_input(fq, port_idx)
        set_node_value(_node, value)

    _update_node_val(1, scale)


#    _update_node_val(2, max_level)
#    _update_node_val(3, output_low)
#    _update_node_val(4, output_high)


def compute_stats_layouts(config, model, qscheme=None):
    """
    Compute stats layouts and hardware configuration
    :param config: dictionary with params algo section from toolkit config
    :param model: CompressedModel instance
    :return: configuration dictionary
    """
    hardware_config = load_hardware_config(config)
    fq_configuration = \
        read_all_fake_quantize_configurations(config, hardware_config, model)
    if not config.preset:
        config.preset = 'performance'
    if not qscheme:
        fq_configuration = get_configurations_by_preset(config, model, fq_configuration, hardware_config)
        fq_configuration = add_range_estimator_configs(fq_configuration, config)
    else:
        fq_configuration = get_configurations_by_qscheme(fq_configuration, qscheme)

    change_configurations_by_model_type(model, config, fq_configuration, hardware_config)

    # get all fake quantize nodes
    fq_nodes = get_nodes_by_type(model, ['ConvertFP8'])

    fake_quantize_config = {}
    for fq in fq_nodes:
        is_weights = fq['fq_group'] == 'weights'
        fq_config = copy(fq_configuration[fq.name][fq['fq_group']])
        fake_quantize_config[fq.fullname] = fq_config
        if fq.fullname in config.layerwise_configs[0]:
            fq_config = Dict(merge_nested_dicts(fq_config, config.layerwise_configs[0][fq.fullname]))

        fq_config['signed'] = False
        if 'level_low' in fq_config and 'level_high' in fq_config and fq_config['level_low'] < 0:
            fq_config['signed'] = True

        fake_quantize_config[fq.fullname] = fq_config
        # fq.levels = compute_levels(fq_config, is_weights)

    return fake_quantize_config


def get_value(key, fq_config, default=None):
    return fq_config[key] if key in fq_config else default


def compute_levels(fq_config, is_weights):
    def_levels = 2 ** get_value('bits', fq_config, 8)

    if is_weights and fq_config['mode'] == 'symmetric':
        level_low = get_value('level_low', fq_config, -def_levels / 2 + 1)
    else:
        level_low = get_value('level_low', fq_config, -def_levels / 2)
    level_high = get_value('level_high', fq_config, def_levels / 2 - 1)
    return int(abs(level_high) + abs(level_low) + 1)


def insert_fake_quantize_nodes(config, model, qscheme=None):
    """ Inserts fake quantize nodes, fill them according config
    :param config: dictionary with params algo section from toolkit config
    :param model: CompressedModel instance
    :param qscheme: The quantization scheme generated from the space
    :return None
    """
    hardware_config = load_hardware_config(config)
    ignored_params = {
        'skip_model': False,
        'scope': [],
        'operations': []
    }
    if config['ignored']:
        ignored_params.update(deepcopy(config['ignored']))

    if config['model_type']:
        ignored_params['operations'] += get_ignored_operations(config['model_type'])

    if qscheme:
        for key in qscheme:
            if qscheme[key]['quantize'] == 0 and key not in ignored_params['scope']:
                ignored_params['scope'].append(key)

    GraphTransformer(hardware_config).insert_fake_quantize(model, ignored_params)


def get_fake_quantize_input(fake_quantize):
    """ Returns input into fake quantize node
    :param fake_quantize: fake quantize node
    :return Input node of fake quantize node
    """
    parent = get_node_input(fake_quantize, 0)
    if parent.attrs()['op'] == 'Cast':
        parent = get_node_input(parent, 0)
    return parent


def get_fake_quantize_input_value(fake_quantize):
    """ Returns input into fake quantize node
    :param fake_quantize: fake quantize node
    :return Input node of fake quantize node
    """
    input_node = fake_quantize
    if input_node.attrs()['op'] == 'Cast':
        input_node = get_node_input(input_node, 0)
    return get_input_data_value(input_node, 0)


def get_fake_quantize_first_output(fake_quantize):
    """ Returns first output of the fake quantize node (usually used for weights)
    :param fake_quantize: fake quantize node
    :return metadata of the node which is first output of the fake quantize node
    """
    return get_node_output(fake_quantize, 0)[0]


def fix_zero_filters_symmetric(max_level, eps=0.01):
    max_range = np.max(max_level)
    lower_threshold = np.maximum(8e-5, eps * max_range)
    return np.maximum(lower_threshold, max_level)


def fix_zero_filters_asymmetric(max_level, min_level, eps=1e-8):
    ranges = max_level - min_level
    ranges = ranges if isinstance(ranges, np.ndarray) else np.array([ranges])
    min_correction = 8 * 10e-5
    corrections = [(np.maximum(eps * rng, rng) - rng) * 0.5 if rng > min_correction
                   else min_correction for rng in ranges]
    max_level = max_level + corrections
    min_level = min_level - corrections
    return max_level, min_level


def symmetric_range(node, fq, weights_stats,
                    batch_inputs_stats, fake_quantize_config):
    name = get_quantized_input_key(fq)
    if node.type == 'Const' or get_input_data_value(fq, 0) is not None:
        node_output = get_fake_quantize_first_output(fq)
        max_level = weights_stats[node_output.fullname]['max']
        max_level = fix_zero_filters_symmetric(max_level)
        min_level = -max_level
        # min_level = weights_stats[node_output.fullname]['min']
        # min_level = fix_zero_filters_symmetric(min_level)
    elif name in batch_inputs_stats:
        max_level = batch_inputs_stats[name]['max']
        min_level = batch_inputs_stats[name]['min']
        max_level = fix_zero_filters_symmetric(max_level)
        signed = fake_quantize_config[fq.fullname]['signed']
        min_level = -max_level  # np.zeros(max_level.shape) if np.all(min_level >= 0) and not signed else \
        # -max_level * fq.levels / (fq.levels - 2)
    else:
        raise Exception(
            'WARNING: Fake quantize node {} is missed'.format(fq.fullname))
    min_level, max_level = broadcast_fq_values(fq, node, min_level, max_level, fake_quantize_config)
    return min_level, max_level


def asymmetric_range(node, fq, weights_stats,
                     batch_inputs_stats, fake_quantize_config, unify_zp=True):
    name = get_quantized_input_key(fq)
    if node.type == 'Const' or get_input_data_value(fq, 0) is not None:
        node_output = get_fake_quantize_first_output(fq)
        max_level = weights_stats[node_output.fullname]['max']
        min_level = weights_stats[node_output.fullname]['min']
    elif name in batch_inputs_stats:
        max_level = batch_inputs_stats[name]['max']
        min_level = batch_inputs_stats[name]['min']
    else:
        raise Exception(
            'WARNING: Fake quantize node {} is missed'.format(fq.fullname))

    max_level, min_level = fix_zero_filters_asymmetric(max_level, min_level)
    min_level = np.where(min_level < 0.0, min_level, 0.0)
    max_level = np.where(max_level > 0.0, max_level, 0.0)
    if unify_zp:
        if name in batch_inputs_stats:
            raise Exception(
                'WARING: unify zero point of fake quantize node {} not supported'.format(fq.fullname)
            )
        min_level, max_level = tune_range_unify_zp(
            min_level, max_level, fake_quantize_config[fq.fullname]['bits'])
    else:
        min_level, max_level = tune_range(
            min_level, max_level, fake_quantize_config[fq.fullname]['bits'])

    min_level, max_level = broadcast_fq_values(fq, node, min_level, max_level, fake_quantize_config)
    return min_level, max_level


def get_quantized_model(model, create_stats_collector, activations_statistics,
                        fill_fq_range, config, qscheme=None):
    """
    Returns a calibrated low precision model via four steps:
    1. Quantize the model
    2. Calculate quantization config for FQ nodes
    3. Collect the weight stats based on config
    4. Calibrate [min, max] for inserted fq nodes
    :param model: original model (CompressedModel instance)
    :param create_stats_collector: functor to create function for stats collector callback
    :param activations_statistics: precomputed statistics for activations layers
    :param fill_fq_range: functor to generate min and max range for fake quantize node
    :param config: dictionary with params algo section from toolkit config
     """
    # ConvertFP8 nodes insertion
    insert_fake_quantize_nodes(config, model, qscheme=qscheme)

    fake_quantize_config = compute_stats_layouts(config, model, qscheme=qscheme)

    # generate a list of fq nodes that require rescaling (first convolutions weight FQs)
    fake_quantize_config.update(set_rescaling_factors(config, model))

    weights_stats_layout = create_stats_collector(fake_quantize_config, model, for_weights=True)

    # compute weights statistics
    weights_stats = compute_weights_stats(model, weights_stats_layout)

    # calculate and fill min and max range for fq nodes
    fill_fq_range(model, weights_stats, activations_statistics, fake_quantize_config, config)
    return model


def compute_weights_stats(model, stats_layout):
    """ Computes weights statistic from provided statistics layout
    :param model: CompressedModel instance
    :param stats_layout: dictionary with layer names as keys and
     functions list with rules how to compute statistics as values
    :return dictionary with layers names as keys and list of evaluated statistics as values"""
    # compute weights statistics
    weights_stats = {}
    for fq_name, stats in stats_layout.items():
        fq_node = get_node_by_name(model, fq_name)
        if fq_node.type != 'ConvertFP8':
            raise Exception('ConvertFP8 node for weights is missed')
        node = get_fake_quantize_first_output(fq_node)
        weights_node = get_node_input(fq_node, 0)
        weights_value = get_input_data_value(fq_node, 0)
        if weights_node.type != 'Const' and weights_value is None:
            raise Exception('Incorrect stats layout for weights:'
                            ' {} is activation'.format(weights_node.name))
        if node.fullname not in weights_stats:
            weights_stats[node.fullname] = {}
        for stat_name, stat_fn in stats.items():
            weights = weights_value.astype(np.float32)
            weights_stats[node.fullname][stat_name] = stat_fn(weights)
    return weights_stats


def broadcast_fq_values(fq, node, min_level, max_level, fq_config):
    """ Reshapes weights and activations in perchannel mode for next fusing
    :param fq: current Fake Quantize node
    :param node: input node for Fake Quantize
    :param min_level:
    :param max_level:
    :param fq_config: for checking special Fake Quantize
    :return tuple of reshaped min and max values"""

    min_level = np.array(min_level)
    max_level = np.array(max_level)

    if not min_level.shape and not max_level.shape:
        return min_level, max_level

    # get input shape from data node
    input_shape = get_input_shape(fq, 0)
    bounds_shape = np.ones(len(input_shape), dtype=np.int32)

    if node.type == 'Const':
        output_node = get_fake_quantize_first_output(fq)
        if output_node.type in [op['type'] for op in TRANSPOSED_OPERATIONS]:
            bounds_shape[1] = input_shape[1]
        else:
            bounds_shape[0] = input_shape[0]
    else:
        if fq_config[fq.fullname]['granularity'] == 'perchannel':
            bounds_shape[1] = input_shape[1]

    min_level = min_level.reshape(bounds_shape)
    max_level = max_level.reshape(bounds_shape)

    return min_level, max_level


def set_rescaling_factors(config, model, scaling_factor=2.0):
    """
        Generate a list of weight FQ nodes for input convolutions
        for further rescaling of weights/FQs.
        Skip if target device is not CPU.
        :param config: algo config
        :param model: CompressedModel instance
        :param scaling_factor: rescaling factor for first convolution nodes
    """

    fqs_to_rescale = []
    #saturation_fix = config.get('saturation_fix', 'no')
    saturation_fix = 'no'

    if config['target_device'] not in ['CPU', 'ANY'] \
            or not get_nodes_by_type(model, ['Convolution'], recursively=False) \
            or saturation_fix == 'no':
        return {'scaling_factor': 1.0,
                'fqs_to_rescale': fqs_to_rescale}

    input_nodes = get_nodes_by_type(model, ['Parameter'], recursively=False)
    fc_layers = get_nodes_by_type(model, [op['type'] for op in OPERATIONS_WITH_WEIGHTS], recursively=False)

    fc_layers_to_rescale = []
    if saturation_fix == 'first_layer':
        fc_layers_to_rescale = get_first_convolutions(input_nodes)
    elif saturation_fix == 'all':
        fc_layers_to_rescale = fc_layers

    for node in fc_layers_to_rescale:
        fqs_to_rescale.append(get_node_input(node, 1).name)

    for fc_layer in fc_layers:
        fq_input_name = get_node_input(fc_layer, 1).name
        if 'need_rescale' in fc_layer and fc_layer['need_rescale'] and \
                fq_input_name not in fqs_to_rescale:
            fqs_to_rescale.append(fq_input_name)

    return {'scaling_factor': scaling_factor,
            'fqs_to_rescale': fqs_to_rescale}


def unify_fq_scales(model, config):
    def _custom_broadcast(arrays_list):
        arrays_list = np.broadcast_arrays(*list(arr.T for arr in arrays_list))
        return [arr.T for arr in arrays_list]

    for _, fqs in find_fqs_to_unify(model, config):
        min_levels = []
        max_levels = []
        for fq in fqs:
            fq = get_node_by_name(model, fq)
            fq_inputs = get_node_inputs(fq)[1:]
            min_levels.append(get_node_value(fq_inputs[0]))
            max_levels.append(get_node_value(fq_inputs[1]))
        orig_shapes = [s.shape for s in min_levels]
        min_levels = _custom_broadcast(min_levels)
        max_levels = _custom_broadcast(max_levels)
        for i, fq in enumerate(fqs):
            fq = get_node_by_name(model, fq)
            min_level = np.min(min_levels, axis=0).reshape(orig_shapes[i])
            max_level = np.max(max_levels, axis=0).reshape(orig_shapes[i])

            fill_fake_quantize_node(fq, min_level, max_level)


def create_renamed_layers_mapping(model, stats_layout):
    changed_names_map = {}
    for layer_name in stats_layout:
        node_name = layer_name
        port_id = None
        if isinstance(layer_name, tuple):
            node_name, port_id = layer_name
        node = get_node_by_name(model, node_name)
        if node is not None and 'orig_node_name' in node:
            name_change_to = node['orig_node_name'] if port_id is None else (node['orig_node_name'], port_id)
            changed_names_map[layer_name] = name_change_to
    return changed_names_map


def get_num_levels(x: np.ndarray) -> int:
    """
        Calculates the number of discret levels of the values
        in the input NumPy tensor x
        :param x: the input tensor
        :return the number of discret value levels in the input tensor x
    """
    NUM_BINS = 256
    x = x.flatten()
    hist, _ = np.histogram(x, NUM_BINS)
    non_empty_bins = [i for i, v in enumerate(hist) if v > 0]
    deltas = [non_empty_bins[i] - non_empty_bins[i - 1] for i in range(1, len(non_empty_bins))]
    if deltas == []:
        return 0
    d = min(deltas)
    if d == 1:
        return -1

    return round(NUM_BINS / d)

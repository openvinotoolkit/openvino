# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino.tools.pot.statistics.function_selector import AGGREGATION_FN, ACTIVATIONS_STATS_FN, WEIGHTS_STATS_FN, \
    get_aggregation_function, get_stats_function_for_activations, get_stats_function_for_weights, PERCHANNEL, PERTENSOR

from openvino.tools.pot.algorithms.quantization.fake_quantize import get_num_levels

INPUT_SHAPES = [(2, 2, 1), (2, 2, 2)]
AGG_INPUTS = [np.reshape(np.array(range(np.prod(shape)), dtype=np.float32), shape) for shape in INPUT_SHAPES]

GOLD_VALUES_AGGREGATION_FUNCTIONS_2_2_1 = {
    'mean': 1.5,
    'max': 3.,
    'min': 0.,
    'median': 1.5,
    'mean_no_outliers': 1.5,
    'median_no_outliers': 1.5,
    'hl_estimator': 1.5,
    'batch_mean': np.array([[[1], [2]]], dtype=np.float)
}

GOLD_VALUES_AGGREGATION_FUNCTIONS_2_2_2 = {
    'mean': [3., 4.],
    'max': [6., 7.],
    'min': [0., 1.],
    'median': [3., 4.],
    'mean_no_outliers': [3., 4.],
    'median_no_outliers': [3., 4.],
    'hl_estimator': [3., 4.],
    'batch_mean': np.array([[[2, 3], [4, 5]]], dtype=np.float)
}

GOLD_VALUES_AGGREGATION_FUNCTIONS = [
    GOLD_VALUES_AGGREGATION_FUNCTIONS_2_2_1,
    GOLD_VALUES_AGGREGATION_FUNCTIONS_2_2_2
]


@pytest.mark.parametrize(
    'name', AGGREGATION_FN.registry_dict.keys(),
    ids=list(AGGREGATION_FN.registry_dict.keys()))
def test_aggregation_function(name):
    for i, input_tensor in enumerate(AGG_INPUTS):
        fn = get_aggregation_function(name)
        result = fn(input_tensor)
        expected = GOLD_VALUES_AGGREGATION_FUNCTIONS[i][name]
        np.testing.assert_almost_equal(result, expected)


INPUT_SHAPE = (2, 2, 2, 2)
INPUT = np.reshape(np.array(range(np.prod(INPUT_SHAPE)), dtype=np.float32), INPUT_SHAPE)

GOLD_VALUES_ACTIVATION_FUNCTIONS = {
    'perchannel': {
        'min': [[0., 4.], [8., 12.]],
        'max': [[3., 7.], [11., 15.]],
        'abs_max': [[3., 7.], [11., 15.]],
        'quantile': [[0.03, 4.03], [8.03, 12.03]],
        'abs_quantile': [[0.03, 4.03], [8.03, 12.03]],
        'mean': [[1.5, 5.5], [9.5, 13.5]],
        'mean_axis': [[1.5, 5.5], [9.5, 13.5]]
    },
    'pertensor': {
        'min': [0., 8.],
        'max': [7., 15.],
        'abs_max': [7., 15.],
        'quantile': [0.07, 8.07],
        'abs_quantile': [0.07, 8.07]
    }
}

ACTIVATIONS_STATS_FUNCTIONS = [
    *[(name, PERTENSOR) for name in ACTIVATIONS_STATS_FN[PERTENSOR].registry_dict.keys()],
    *[(name, PERCHANNEL) for name in ACTIVATIONS_STATS_FN[PERCHANNEL].registry_dict.keys()],
]


@pytest.mark.parametrize(
    'name, scale', ACTIVATIONS_STATS_FUNCTIONS,
    ids=['{}_{}'.format(fn[0], fn[1]) for fn in ACTIVATIONS_STATS_FUNCTIONS])
def test_activation_function(name, scale):
    fn = get_stats_function_for_activations(name, scale, 'compute_statistic')
    if name in ['quantile', 'abs_quantile']:
        result = fn(INPUT, q=1e-2)
    else:
        result = fn(INPUT)
    expected = GOLD_VALUES_ACTIVATION_FUNCTIONS[scale][name]
    np.testing.assert_almost_equal(result, expected)


GOLD_VALUES_WEIGHT_FUNCTIONS = {
    'perchannel': {
        'min': [0., 8.],
        'max': [7., 15.],
        'abs_max': [7., 15.],
        'quantile': [0.07, 8.07],
        'abs_quantile': [0.07, 8.07]
    },
    'pertensor': {
        'min': 0.,
        'max': 15.,
        'abs_max': 15.,
        'quantile': 0.15,
        'abs_quantile': 0.15
    }
}

WEIGHTS_STATS_FUNCTIONS = [
    *[(name, PERTENSOR) for name in WEIGHTS_STATS_FN[PERTENSOR].registry_dict.keys()],
    *[(name, PERCHANNEL) for name in WEIGHTS_STATS_FN[PERCHANNEL].registry_dict.keys()],
]


@pytest.mark.parametrize(
    'name, scale', WEIGHTS_STATS_FUNCTIONS,
    ids=['{}_{}'.format(fn[0], fn[1]) for fn in WEIGHTS_STATS_FUNCTIONS])
def test_weights_function(name, scale):
    fn = get_stats_function_for_weights(name, scale)
    if name in ['quantile', 'abs_quantile']:
        result = fn(INPUT, q=1e-2)
    else:
        result = fn(INPUT)
    expected = GOLD_VALUES_WEIGHT_FUNCTIONS[scale][name]
    np.testing.assert_almost_equal(result, expected)


GOLD_VALUES_CH_TRANS_WEIGHT_FUNCTIONS = {
    'min': [0., 4.],
    'max': [11., 15.],
    'abs_max': [11., 15.],
    'quantile': [0.07, 4.07],
    'abs_quantile': [0.07, 4.07]
}
WEIGHTS_CH_STATS_FUNCTIONS = [(name, True) for name in
                              WEIGHTS_STATS_FN[PERCHANNEL].registry_dict.keys()]


NUM_LEVELS_PARAMS = [
    (np.random.randint, (0, 2, (3, 100, 100)), 1, 1),
    (np.random.randint, (-32, 32, (3, 100, 100)), 64, 1),
    (np.random.randint, (-32, 32, (3, 100, 100)), 64, 1/512),
    (np.random.rand, (3, 100, 100), -1, 1),
    (np.random.randint, (0, 1, (3, 100, 100)), 0, 1)
]


@pytest.mark.parametrize('gen_func,params,expected,coef', NUM_LEVELS_PARAMS)
def test_get_num_levels_function(gen_func, params, expected, coef):
    test_1 = gen_func(*params) * coef
    result = get_num_levels(test_1)
    assert result == expected

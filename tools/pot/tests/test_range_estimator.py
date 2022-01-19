# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from addict import Dict

from openvino.tools.pot.algorithms.quantization.fake_quantize import compute_stats_layouts
from openvino.tools.pot.algorithms.quantization.minmax.algorithm import MinMaxQuantization
from openvino.tools.pot.graph import load_model
from openvino.tools.pot.statistics.functions import activations as asf
from openvino.tools.pot.statistics.functions import weights as wsf

GOLD_VALUES = [
    {
        'weights': {
            'max': wsf.abs_max_per_filter
        },
        'activations': {
            'min': asf.min_per_tensor,
            'max': asf.abs_max_per_tensor
        }
    },
    {
        'weights': {
            'min': wsf.quantile_per_filter,
            'max': wsf.quantile_per_filter
        },
        'activations': {
            'min': asf.quantile_per_tensor,
            'max': asf.quantile_per_tensor
        }
    }
]

TEST_CONFIG = [
    ('mobilenet-v2-pytorch', 'pytorch', 'symmetric', 'default', GOLD_VALUES[0]),
    ('mobilenet-v2-pytorch', 'pytorch', 'asymmetric', 'quantile', GOLD_VALUES[1]),
]


def get_algo_config(quantization_mode, range_estimator_preset):
    return Dict({
        'name': 'MinMaxQuantization',
        'target_device': 'CPU',
        'preset': 'accuracy',
        'stat_subset_size': 1000,
        'weights': {
            'bits': 8,
            'mode': quantization_mode,
            'granularity': 'perchannel',
            'range_estimator': {
                'preset': range_estimator_preset
            }
        },
        'activations': {
            'bits': 8,
            'mode': quantization_mode,
            'granularity': 'pertensor',
            'range_estimator': {
                'preset': range_estimator_preset
            }
        }
    })


@pytest.mark.parametrize(
    'model_name, model_framework, quantization_mode, range_estimator_preset, expected_fns', TEST_CONFIG,
    ids=['{}_{}_{}_{}'.format(v[0], v[1], v[2], v[3]) for v in TEST_CONFIG])
def test_range_estimator(tmp_path, models, model_name,
                         model_framework, quantization_mode,
                         range_estimator_preset, expected_fns):
    def check_statistics_layout(stats_layout, for_weights):
        tensor_type = 'weights' if for_weights else 'activations'
        for stats in stats_layout.values():
            assert len(expected_fns[tensor_type]) == len(stats)
            for stats_name, fn in stats.items():
                assert stats_name in ['min', 'max']
                if hasattr(fn, 'func'):
                    fn = fn.func
                assert fn == expected_fns[tensor_type][stats_name]

    algo_config = get_algo_config(quantization_mode, range_estimator_preset)

    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)

    fake_quantize_config = compute_stats_layouts(algo_config, model)

    weights_stats_layout = MinMaxQuantization.create_stats_layout(
        fake_quantize_config, model, for_weights=True)
    check_statistics_layout(weights_stats_layout, for_weights=True)

    act_stats_layout = MinMaxQuantization.create_stats_layout(
        fake_quantize_config, model, for_weights=False)
    check_statistics_layout(act_stats_layout, for_weights=False)

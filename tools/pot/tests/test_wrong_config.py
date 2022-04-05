# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from openvino.tools.pot.configs.config import Config
from .utils.path import TOOL_CONFIG_PATH

ALGORITHM_SETTINGS = {
    'wrong_preset': (
        {
            'name': 'MinMaxQuantization',
            'params': {
                'perset': 'accuracy',
                'stat_subset_size': 1
            }
        },
        'Algorithm MinMaxQuantization. Unknown parameter: perset'
    ),
    'wrong_stats_subset_size': (
        {
            'name': 'DefaultQuantization',
            'params': {
                'preset': 'accuracy',
                'stats_subset_size': 1
            }
        },
        'Algorithm DefaultQuantization. Unknown parameter: stats_subset_size'
    ),
    'wrong_weights': (
        {
            'name': 'DefaultQuantization',
            'params': {
                'activations': {
                    'bits': 8,
                    'mode': 'symmetric',
                    'granularity': 'pertensor',
                    'range_estimator': {
                        'preset': 'quantile'
                    }
                },
                'weight': {
                    'bits': 8,
                    'level_low': -127,
                    'level_high': 127
                },
                'stat_subset_size': 1
            }
        },
        'Algorithm DefaultQuantization. Unknown parameter: weight'
    ),
    'wrong_mode': (
        {
            'name': 'DefaultQuantization',
            'params': {
                'activations': {
                    'bits': 8,
                    'type': 'symmetric',
                    'granularity': 'pertensor',
                    'range_estimator': {
                        'preset': 'quantile'
                    }
                },
                'weights': {
                    'bits': 8,
                    'level_low': -127,
                    'level_high': 127
                },
                'stat_subset_size': 1
            }
        },
        'Algorithm DefaultQuantization. Unknown parameter: type'
    ),
    'wrong_outlier_prob': (
        {
            'name': 'AccuracyAwareQuantization',
            'params': {
                'metric_subset_ratio': 0.5,
                'ranking_subset_size': 300,
                'max_iter_num': 10,
                'maximal_drop': 0.005,
                'drop_type': 'absolute',
                'use_prev_if_drop_increase': False,
                'base_algorithm': 'DefaultQuantization',
                'activations': {
                    'bits': 8,
                    'mode': 'symmetric',
                    'granularity': 'pertensor',
                    'range_estimator': {
                        'preset': 'quantile'
                    }
                },
                'weights': {
                    'bits': 8,
                    'level_low': -127,
                    'level_high': 127,
                    'range_estimator': {
                        'max': {
                            'type': 'quantile',
                            'outlier': 0.0001
                        }
                    }
                },
                'stat_subset_size': 1
            }
        },
        'Algorithm AccuracyAwareQuantization. Unknown parameter: outlier'
    ),
    'wrong_maximal_drop': (
        {
            'name': 'AccuracyAwareQuantization',
            'params': {
                'metric_subset_ratio': 0.5,
                'ranking_subset_size': 300,
                'max_iter_num': 10,
                'max_drop': 0.005,
                'drop_type': 'absolute',
                'use_prev_if_drop_increase': False,
                'base_algorithm': 'DefaultQuantization',
                'activations': {
                    'bits': 8,
                    'mode': 'symmetric',
                    'granularity': 'pertensor',
                    'range_estimator': {
                        'preset': 'quantile'
                    }
                },
                'weights': {
                    'bits': 8,
                    'level_low': -127,
                    'level_high': 127,
                    'range_estimator': {
                        'max': {
                            'type': 'quantile',
                            'outlier_prob': 0.0001
                        }
                    }
                },
                'stat_subset_size': 1
            }
        },
        'Algorithm AccuracyAwareQuantization. Unknown parameter: max_drop'
    ),
    'wrong_algo_keys': (
        {
            "name": "FastBiasCorrection",
            "stat_subset_size": 10,
            "target_device": "ANY",
        },
        'Unsupported params for FastBiasCorrection algorithm section: stat_subset_size, target_device'
    )
}


@pytest.mark.parametrize(
    'algorithm_settings', ALGORITHM_SETTINGS.items(),
    ids=['{}_config'.format(os.path.splitext(c)[0]) for c in ALGORITHM_SETTINGS]
)
def test_algo_params_validation(algorithm_settings):
    tool_config_path = TOOL_CONFIG_PATH.joinpath('mobilenet-v2-pytorch_single_dataset.json').as_posix()
    config = Config.read_config(tool_config_path)
    config['compression']['algorithms'][0] = algorithm_settings[1][0]
    config_error = algorithm_settings[1][1]

    with pytest.raises(RuntimeError, match=config_error):
        config.validate_algo_config()

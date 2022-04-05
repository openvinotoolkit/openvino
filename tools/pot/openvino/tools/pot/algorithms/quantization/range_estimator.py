# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy


DEFAULT_ACTIVATIONS_RANGE_ESTIMATOR_CONFIG = {
    'perchannel': {
        'symmetric': {
            'min': {'aggregator': 'min', 'type': 'min', 'granularity': 'pertensor'},
            'max': {'aggregator': 'max', 'type': 'abs_max'}
        },
        'asymmetric': {
            'min': {'aggregator': 'min', 'type': 'min'},
            'max': {'aggregator': 'max', 'type': 'max'}
        }
    },
    'pertensor': {
        'symmetric': {
            'min': {'aggregator': 'min', 'type': 'min'},
            'max': {'aggregator': 'mean', 'type': 'abs_max'}
        },
        'asymmetric': {
            'min': {'aggregator': 'mean', 'type': 'min'},
            'max': {'aggregator': 'mean', 'type': 'max'}
        }
    }}


QUANTILE_ACTIVATIONS_RANGE_ESTIMATOR_CONFIG = {
    'perchannel': {
        'symmetric': {
            'min': {'aggregator': 'min', 'type': 'min', 'granularity': 'pertensor'},
            'max': {'aggregator': 'max', 'type': 'abs_quantile', 'outlier_prob': 1e-4}
        },
        'asymmetric': {
            'min': {'aggregator': 'min', 'type': 'quantile', 'outlier_prob': 1e-4},
            'max': {'aggregator': 'max', 'type': 'quantile', 'outlier_prob': 1e-4}
        }
    },
    'pertensor': {
        'symmetric': {
            'min': {'aggregator': 'min', 'type': 'min'},
            'max': {'aggregator': 'mean', 'type': 'abs_quantile', 'outlier_prob': 1e-4}
        },
        'asymmetric': {
            'min': {'aggregator': 'mean', 'type': 'quantile', 'outlier_prob': 1e-4},
            'max': {'aggregator': 'mean', 'type': 'quantile', 'outlier_prob': 1e-4}
        }
    }}


DEFAULT_WEIGHTS_RANGE_ESTIMATOR_CONFIG = {
    'symmetric': {
        'max': {'type': 'abs_max'}
    },
    'asymmetric': {
        'min': {'type': 'min'},
        'max': {'type': 'max'}
    }}


QUANTILE_WEIGHTS_RANGE_ESTIMATOR_CONFIG = {
    'symmetric': {
        'max': {'type': 'abs_quantile', 'outlier_prob': 1e-4}
    },
    'asymmetric': {
        'min': {'type': 'quantile', 'outlier_prob': 1e-4},
        'max': {'type': 'quantile', 'outlier_prob': 1e-4}
    }}


RANGE_ESTIMATOR_CONFIG_PRESETS = {
    'default': {
        'activations': DEFAULT_ACTIVATIONS_RANGE_ESTIMATOR_CONFIG,
        'weights': DEFAULT_WEIGHTS_RANGE_ESTIMATOR_CONFIG,
    },
    'quantile': {
        'activations': QUANTILE_ACTIVATIONS_RANGE_ESTIMATOR_CONFIG,
        'weights': QUANTILE_WEIGHTS_RANGE_ESTIMATOR_CONFIG,
    }
}


def get_range_estimator_config(config, tensor_type, granularity, q_mode, preset=None):
    global_range_estimator = config.get('range_estimator', {})
    tensor_type_config = config.get(tensor_type, {})
    range_estimator = tensor_type_config.get('range_estimator', global_range_estimator)
    if preset is None:
        preset = range_estimator.get('preset', 'default')
    preset_config = deepcopy(RANGE_ESTIMATOR_CONFIG_PRESETS[preset][tensor_type])
    result_config = preset_config[granularity][q_mode] \
        if tensor_type == 'activations' else preset_config[q_mode]
    if 'min' in range_estimator:
        if 'min' not in result_config:
            result_config['min'] = {}
        result_config['min'].update(range_estimator['min'])
    if 'max' in range_estimator:
        if 'max' not in result_config:
            result_config['max'] = {}
        result_config['max'].update(range_estimator['max'])
    for fn_config in result_config.values():
        if 'granularity' not in fn_config:
            fn_config['granularity'] = granularity
    return result_config

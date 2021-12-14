# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from addict import Dict

from openvino.tools.pot.engines.ac_engine import ACEngine
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.engines.utils import append_stats
from openvino.tools.pot.statistics.statistics import TensorStatistic


def test_ac_engine_append_stats():
    config = _get_accuracy_checker_config()
    engine = ACEngine(config)
    run_append_stats_test(engine)


def test_ie_engine_append_stats():
    engine = IEEngine(Dict({"device": "CPU"}), None, None)
    run_append_stats_test(engine)


def run_append_stats_test(engine):
    # pylint: disable=W0212
    sample_tensor = np.random.uniform(0, 1, (10, 10))
    stat_fn = stat_name = TensorStatistic(lambda tensor: tensor)
    stats_layout = {
        'conv_layer': {stat_name: stat_fn},
        'fc_layer': {stat_name: stat_fn},
    }
    for name, _ in stats_layout.items():
        stats_layout[name][stat_fn].kwargs = {}
    value = {'conv_layer': sample_tensor, 'fc_layer': sample_tensor}
    append_stats(engine._accumulated_layer_stats, stats_layout, value, dataset_index=0)
    for layer, accumulated_value in engine._accumulated_layer_stats.items():
        assert np.array_equal(accumulated_value[stat_name][0][1], value[layer])

    engine._accumulated_layer_stats = {}
    value = [
        {'conv_layer': sample_tensor, 'fc_layer': sample_tensor},
        {'conv_layer': sample_tensor, 'fc_layer': sample_tensor},
    ]
    append_stats(engine._accumulated_layer_stats, stats_layout, value, dataset_index=0)
    for layer, accumulated_value in engine._accumulated_layer_stats.items():
        assert np.array_equal(
            accumulated_value[stat_name][0][1][:, 0], value[0][layer]
        )


def _get_accuracy_checker_config():
    return Dict(
        {
            'models': [
                {
                    'launchers': [{'framework': 'dlsdk', 'device': 'CPU'}],
                    'datasets': [{'data_source': '.'}],
                }
            ]
        }
    )

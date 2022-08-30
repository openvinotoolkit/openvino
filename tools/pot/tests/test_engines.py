# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

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


def create_ng_mock(return_value=None):
    ng_const_out_mock = Mock()
    ng_tensor_desc_mock = Mock()
    ng_const_out_mock.get_tensor.return_value = ng_tensor_desc_mock
    ng_tensor_desc_mock.get_names.return_value = return_value
    return ng_const_out_mock


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
    conv_layer_mock = create_ng_mock(['conv_layer'])
    fc_layer_mock = create_ng_mock(['fc_layer'])
    value = {conv_layer_mock: sample_tensor, fc_layer_mock: sample_tensor}
    ref_value = {'conv_layer': sample_tensor, 'fc_layer': sample_tensor}
    append_stats(engine._accumulated_layer_stats, stats_layout, value, dataset_index=0)
    for layer, accumulated_value in engine._accumulated_layer_stats.items():
        assert np.array_equal(accumulated_value[stat_name][0][1], ref_value[layer])

    engine._accumulated_layer_stats = {}
    value = [
        {conv_layer_mock: sample_tensor, fc_layer_mock: sample_tensor},
        {conv_layer_mock: sample_tensor, fc_layer_mock: sample_tensor},
    ]
    ref_value = [
        {'conv_layer': sample_tensor, 'fc_layer': sample_tensor},
        {'conv_layer': sample_tensor, 'fc_layer': sample_tensor},
    ]
    append_stats(engine._accumulated_layer_stats, stats_layout, value, dataset_index=0)
    for layer, accumulated_value in engine._accumulated_layer_stats.items():
        assert np.array_equal(
            accumulated_value[stat_name][0][1][:, 0], ref_value[0][layer]
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

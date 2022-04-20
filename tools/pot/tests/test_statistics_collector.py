# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os

from pathlib import Path
import pytest
import numpy as np
from addict import Dict

from openvino.tools.pot.graph import load_model
from openvino.tools.pot.data_loaders.creator import create_data_loader
from openvino.tools.pot.engines.creator import create_engine
from openvino.tools.pot.statistics.collector import StatisticsCollector
from openvino.tools.pot.algorithms.quantization.minmax.algorithm import MinMaxQuantization
from openvino.tools.pot.algorithms.quantization.bias_correction.algorithm import BiasCorrection
from .utils.config import PATHS2DATASETS_CONFIG

TEST_MODELS = [('mobilenet-v2-pytorch', 'pytorch'), ('lstm_outs_quantization', 'tf')]

@pytest.mark.parametrize(
    'model_name, model_framework', TEST_MODELS,
    ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MODELS])
def test_statistics_collector_subsets(tmp_path, models, model_name, model_framework):
    with open(PATHS2DATASETS_CONFIG.as_posix()) as f:
        data_source = Dict(json.load(f))['ImageNet2012'].pop('source_dir')

    engine_config = Dict({'type': 'simplified',
                          'data_source': '{}/{}'.format(data_source, 'ILSVRC2012_val*'),
                          'device': 'CPU'})

    minmax_config = Dict({
        'target_device': 'CPU',
        'preset': 'performance',
        'stat_subset_size': 1,
        'ignored': []
    })
    bias_correction_config = Dict({
        'target_device': 'CPU',
        'preset': 'performance',
        'stat_subset_size': 2
    })

    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)
    data_loader = create_data_loader(engine_config, model)
    engine = create_engine(engine_config, data_loader=data_loader, metric=None)
    collector = StatisticsCollector(engine)
    min_max_algo = MinMaxQuantization(minmax_config, engine)
    min_max_algo.register_statistics(model, collector)
    bias_correction_algo = BiasCorrection(bias_correction_config, engine)
    bias_correction_algo.register_statistics(model, collector)
    collector.compute_statistics(model)

    out = {'MinMaxQuantization': collector.get_statistics_for_algorithm('MinMaxQuantization'),
           'BiasCorrection': collector.get_statistics_for_algorithm('BiasCorrection')}

    refs_file = Path(__file__).parent / 'data/test_cases_refs' / f'{model_name}_statistics_data.json'
    local_path = os.path.join(tmp_path, '{}_{}.json'.format(model_name, 'statistics_data'))
    local_file = open(local_path, 'w')

    with open(refs_file.as_posix()) as file:
        refs = json.load(file)

    eps = 1e-3
    local_out = {}
    for algo_name, algo_val in out.items():
        local_out[algo_name] = {}
        for node_name, node_val in algo_val.items():
            if isinstance(node_name, tuple):
                name = f'{node_name[0]}.{node_name[1]}'
            else:
                name = node_name
            local_out[algo_name][name] = {}
            for stats_name, stats_val in node_val.items():
                local_out[algo_name][name][stats_name] = [np.array(v).tolist() for v in stats_val]
    json.dump(local_out, local_file)
    for algo_name, algo_val in out.items():
        for node_name, node_val in algo_val.items():
            for stats_name, stats_val in node_val.items():
                if stats_name in ['batch_mean_param_in', 'shape']:
                    continue
                if isinstance(node_name, tuple):
                    node_name = f'{node_name[0]}.{node_name[1]}'
                ref_stats_vals = refs[algo_name][node_name][stats_name]
                for ref_vals, vals in zip(ref_stats_vals, stats_val):
                    assert np.max(np.abs(np.array(ref_vals) - vals)) < eps

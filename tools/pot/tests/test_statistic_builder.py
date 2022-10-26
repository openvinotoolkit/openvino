# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from addict import Dict

import pytest

from openvino.tools.pot.graph import load_model
from openvino.tools.pot.data_loaders.creator import create_data_loader
from openvino.tools.pot.engines.creator import create_engine
from openvino.tools.pot.statistics.collector import StatisticsCollector
from openvino.tools.pot.algorithms.quantization.minmax.algorithm import MinMaxQuantization
from openvino.tools.pot.algorithms.quantization.fast_bias_correction.algorithm import FastBiasCorrection
from openvino.tools.pot.algorithms.quantization.bias_correction.algorithm import BiasCorrection
from openvino.tools.pot.algorithms.quantization.channel_alignment.algorithm import ActivationChannelAlignment
from openvino.tools.pot.statistics.utils import merge_stats_by_algo_names
from openvino.tools.pot.statistics.statistic_graph_builder import StatisticGraphBuilder
from tests.utils.config import PATHS2DATASETS_CONFIG
from tests.utils.check_graph import check_model

TEST_MODELS = [
    ('resnet_example', 'pytorch', 'symmetric', True, MinMaxQuantization, 'performance', 'perchannel', 0,
     'max', 'min'),
    ('resnet_example', 'pytorch', 'symmetric', True, MinMaxQuantization, 'performance', 'perchannel', 0,
     'abs_max', 'min'),
    ('resnet_example', 'pytorch', 'symmetric', True, MinMaxQuantization, 'mixed', 'pertensor', 0,
     'abs_max', 'max'),
    ('resnet_example', 'pytorch', 'symmetric', True, MinMaxQuantization, 'mixed', 'pertensor', 23,
     'min', 'quantile'),
    ('resnet_example', 'pytorch', 'symmetric', True, MinMaxQuantization, 'performance', 'perchannel', 23,
     'quantile', 'abs_quantile'),
    ('mobilenetv2_example', 'pytorch', 'symmetric', True, ActivationChannelAlignment, 'mixed',
     'perchannel', 1, None, None),
    ('squeezenet1_1_example', 'pytorch', 'symmetric', True, FastBiasCorrection, 'mixed', 'perchannel', 42,
     None, None),
    ('mobilenetv2_ssd_example', 'pytorch', 'symmetric', True, FastBiasCorrection, 'mixed', 'perchannel', 117,
     None, None),
    ('mobilenet_v3_small_example', 'pytorch', 'symmetric', True, BiasCorrection, 'mixed', 'perchannel', 53,
     None, None)
]


def get_algo_config(quantization_mode, algo, preset, granularity, type_max, type_min):
    return Dict({
        'name': algo,
        'target_device': 'CPU',
        'preset': preset,
        'stat_subset_size': 1000,
        'activations': {
            'bits': 8,
            'mode': quantization_mode,
            'granularity': granularity,
            'range_estimator': {
                "max": {'aggregator': "mean",
                        'type': type_max,
                        'outlier_prob': 0.0001},
                "min": {'aggregator': "mean",
                        'type': type_min,
                        'outlier_prob': 0.0001}
            }
        }
    })


def create_(tmp_path, models, model_name, model_framework, quantization_mode,
            algo, preset, granularity, type_max, type_min
            ):
    with open(PATHS2DATASETS_CONFIG.as_posix()) as f:
        data_source = Dict(json.load(f))['ImageNet2012'].pop('source_dir')

    engine_config = Dict({'type': 'simplified',
                          'data_source': '{}/{}'.format(data_source, 'ILSVRC2012_val*'),
                          'device': 'CPU'})

    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)
    data_loader = create_data_loader(engine_config, model)
    engine = create_engine(engine_config, data_loader=data_loader, metric=None)
    collector = StatisticsCollector(engine)

    algo_config = get_algo_config(quantization_mode, algo, preset, granularity, type_max,
                                  type_min)
    return model, engine, collector, algo_config


# pylint: disable=protected-access,E1305
@pytest.mark.parametrize(
    'model_name, model_framework, quantization_mode, inplace_statistics, \
     algorithm,  preset, granularity, add_output_nodes, type_max, type_min', TEST_MODELS,
    ids=['{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(m[0], m[1], m[2], m[3], m[4].name,
                                             m[5], m[6], m[7], m[8], m[9]) for m in TEST_MODELS])
def test_statistics_collector_subsets(tmp_path, models, model_name, model_framework,
                                      quantization_mode, inplace_statistics, algorithm,
                                      preset, granularity, add_output_nodes, type_max, type_min):
    model, engine, collector, algo_config = create_(tmp_path, models, model_name, model_framework,
                                                    quantization_mode, algorithm.name, preset,
                                                    granularity, type_max, type_min)
    algo = algorithm(algo_config, engine)
    algo._config['inplace_statistics'] = inplace_statistics
    algo.register_statistics(model, collector)
    statistic_graph_builder = StatisticGraphBuilder()
    act_stats_layout, stat_aliases = merge_stats_by_algo_names([algorithm.name], collector._layout_by_algo)
    model_with_nodes, nodes_names, _ = statistic_graph_builder.insert_statistic(model, act_stats_layout, stat_aliases)
    ir_name = f'{model_name}_stat_{type_max}_{type_min}' if type_min is not None \
        else f'{model_name}_stat_mean'
    check_model(tmp_path, model_with_nodes, ir_name, model_framework)
    assert len(set(nodes_names[model.models[0]['model'].name])) == add_output_nodes

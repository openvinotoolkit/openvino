# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
from addict import Dict
from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir

from openvino.tools.pot.app.run import optimize
from openvino.tools.pot import MagnitudeSparsity
from openvino.tools.pot.graph.nx_model import CompressedModel
from openvino.tools.pot.graph.model_utils import get_nodes_by_type
from openvino.tools.pot.graph.node_utils import get_first_convolutions, get_node_value
from openvino.tools.pot.utils.logger import stdout_redirect
from tests.utils.config import get_engine_config, merge_configs
from tests.utils.check_graph import check_graph


# pylint: disable=protected-access
def check_sparsity_level(model, config, ref_sparsity_level):
    """
    Check that sparsity level of the model is equal to reference sparse level.
    """
    if not config.compression.algorithms[0]['params']['normed_threshold']:
        input_nodes = get_nodes_by_type(model, ['Parameter'], recursively=False)
        input_convolutions = get_first_convolutions(input_nodes)
        ignored_scope = [node.fullname for node in input_convolutions]
    else:
        ignored_scope = None

    sparsity_algo = MagnitudeSparsity(config, None, ignored_scope=ignored_scope)
    all_weights_nodes = sparsity_algo._get_all_weights_nodes(model)
    all_weights = [get_node_value(w_node).flatten() for w_node in all_weights_nodes]
    all_weights = np.concatenate(all_weights)
    sparsity_level = np.sum(all_weights == 0) / len(all_weights)
    return np.isclose(sparsity_level, ref_sparsity_level)


TEST_SPARSITY_ALGO = [
    ('sparsity_example', 'tf', 'MagnitudeSparsity', 0.2, False, '_02_sparsity'),
    ('sparsity_example', 'tf', 'MagnitudeSparsity', 0.2, True, '_02_sparsity_normed'),
]


@pytest.mark.parametrize('test_models', TEST_SPARSITY_ALGO,
                         ids=['{}_{}_{}_{}'.format(*m) for m in TEST_SPARSITY_ALGO])
def test_sparsity_algo(test_models, tmp_path, models):
    model_name, model_framework, algorithm, sparsity_level, normed_threshold, ref_name = test_models
    algorithm_config = Dict({
        'algorithms': [{
            'name': algorithm,
            'params': {
                'sparsity_level': sparsity_level,
                'normed_threshold': normed_threshold,
            }
        }]
    })

    model = models.get(model_name, model_framework, tmp_path)

    engine_config = get_engine_config(model_name)
    config = merge_configs(model.model_params, engine_config, algorithm_config)
    config.engine.evaluate = False
    config.engine.type = 'accuracy_checker'

    _ = optimize(config)
    output_dir = os.path.join(config.model.exec_log_dir, 'optimized')
    xml_path = os.path.join(output_dir, config.model.model_name + '.xml')
    bin_path = os.path.join(output_dir, config.model.model_name + '.bin')
    output_model, meta = stdout_redirect(restore_graph_from_ir, xml_path, bin_path)
    output_model.meta_data = meta

    assert check_sparsity_level(CompressedModel(graph=output_model), config, sparsity_level)
    check_graph(tmp_path, output_model, model_name + ref_name, model_framework, check_weights=True)

TEST_SPARSITY_MODELS = [
    ('googlenet-v3', 'tf', 'MagnitudeSparsity', 0.1, False, {'accuracy@top1': 0.8050, 'accuracy@top5': 0.9540}),
    ('googlenet-v3', 'tf', 'MagnitudeSparsity', 0.5, False, {'accuracy@top1': 0.3130, 'accuracy@top5': 0.5020}),
    ('googlenet-v3', 'tf', 'MagnitudeSparsity', 0.5, True, {'accuracy@top1': 0.7090, 'accuracy@top5': 0.8800}),
    ('googlenet-v3', 'tf', 'WeightSparsity', 0.5, True, {'accuracy@top1': 0.769, 'accuracy@top5': 0.937}),
]


@pytest.mark.parametrize('test_models', TEST_SPARSITY_MODELS,
                         ids=['{}_{}_{}_{}'.format(*m) for m in TEST_SPARSITY_MODELS])
def test_sparsity(test_models, tmp_path, models):
    model_name, model_framework, algorithm, sparsity_level, normed_threshold, expected_accuracy = test_models
    algorithm_config = Dict({
        'algorithms': [{
            'name': algorithm,
            'params': {
                'sparsity_level': sparsity_level,
                'normed_threshold': normed_threshold,
            }
        }]
    })

    if algorithm == 'WeightSparsity':
        bias_config = Dict({
            'target_device': 'CPU',
            'stat_subset_size': 300
        })
        algorithm_config['algorithms'][0]['params'].update(bias_config)

    model = models.get(model_name, model_framework, tmp_path)

    engine_config = get_engine_config(model_name)
    config = merge_configs(model.model_params, engine_config, algorithm_config)
    config.engine.models[0].datasets[0].subsample_size = 1000

    metrics = optimize(config)

    output_dir = os.path.join(config.model.exec_log_dir, 'optimized')

    for metric_name in metrics:
        print('{}: {:.4f}'.format(metric_name, metrics[metric_name]))

    assert metrics == pytest.approx(expected_accuracy, abs=0.006)
    xml_path = os.path.join(output_dir, config.model.model_name + '.xml')
    bin_path = os.path.join(output_dir, config.model.model_name + '.bin')
    assert os.path.exists(xml_path)
    assert os.path.exists(bin_path)

    # Check resulting sparsity level
    model, _ = stdout_redirect(restore_graph_from_ir, xml_path, bin_path)
    assert check_sparsity_level(CompressedModel(graph=model), config, sparsity_level)

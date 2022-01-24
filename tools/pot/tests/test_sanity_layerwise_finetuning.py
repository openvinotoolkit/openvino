# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import warnings
import numpy as np
import pytest

from openvino.tools.pot.algorithms.sparsity.default.utils import check_model_sparsity_level
from openvino.tools.pot.data_loaders.creator import create_data_loader
from openvino.tools.pot.engines.creator import create_engine
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.pipeline.initializer import create_pipeline
from tests.utils.check_graph import check_model
from tools.evaluate import evaluate
from .utils.config import get_engine_config, merge_configs, make_algo_config

# pylint: disable=W0611,C0412
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SPARSITY_MODELS = [
    ('mobilenet-v2', 'caffe', 'WeightSparsity', 'performance', 0.3, {'accuracy@top1': 0.3150, 'accuracy@top5': 0.5630})
]


def run_algo(model, model_name, algorithm_config, tmp_path, reference_name):
    engine_config = get_engine_config(model_name)
    config = merge_configs(model.model_params, engine_config, algorithm_config)

    model = load_model(model.model_params)
    data_loader = create_data_loader(engine_config, model)
    engine = create_engine(engine_config, data_loader=data_loader, metric=None)
    pipeline = create_pipeline(algorithm_config.algorithms, engine)

    with torch.backends.mkldnn.flags(enabled=False):
        model = pipeline.run(model)
    paths = save_model(model, tmp_path.as_posix(), reference_name)
    engine.set_model(model)
    metrics = evaluate(config=config, subset=range(1000), paths=paths)
    metrics = OrderedDict([(metric.name, np.mean(metric.evaluated_value))
                           for metric in metrics])

    return metrics, model


@pytest.mark.parametrize('model_params', SPARSITY_MODELS,
                         ids=['{}_{}_sparse_tuning'.format(m[0], m[1]) for m in SPARSITY_MODELS])
def test_sparsity_with_finetuning_algo(models, tmp_path, model_params):
    model_name, model_framework, algo_name, preset, sparsity_level, expected_accuracy = model_params

    if not TORCH_AVAILABLE:
        warnings.warn(UserWarning('Skipping layerwise finetuning test since torch is not importable'))
        return

    additional_params = {
        'sparsity_level': sparsity_level,
        'stat_subset_size': 300,
        'use_layerwise_tuning': True,
        'weights_lr': 1e-5,
        'bias_lr': 1e-3,
        'batch_size': 20,
        'num_samples_for_tuning': 40,
        'tuning_iterations': 1,
        'use_ranking_subset': False,
    }
    algorithm_config = make_algo_config(algo_name, preset, additional_params=additional_params)
    model = models.get(model_name, model_framework, tmp_path)
    reference_name = model_name + '_sparse_tuned'
    metrics, sparse_model = run_algo(model, model_name, algorithm_config, tmp_path, reference_name)
    check_model_sparsity_level(sparse_model, None, sparsity_level, strict=True)
    for metric_name in metrics:
        print('{}: {:.4f}'.format(metric_name, metrics[metric_name]))

    assert metrics == pytest.approx(expected_accuracy, abs=0.006)
    check_model(tmp_path, sparse_model, reference_name,
                model_framework, check_weights=False)


QUANTIZATION_MODELS = [
    ('mobilenet-v2', 'caffe', 'DefaultQuantization', 'performance', {'accuracy@top1': 0.7140, 'accuracy@top5': 0.8970})
]


@pytest.mark.parametrize('model_params', QUANTIZATION_MODELS,
                         ids=['{}_{}_quantize_tuned'.format(m[0], m[1]) for m in QUANTIZATION_MODELS])
def test_quantization_with_finetuning_algo(models, tmp_path, model_params):
    model_name, model_framework, algo_name, preset, expected_accuracy = model_params

    if not TORCH_AVAILABLE:
        warnings.warn(UserWarning('Skipping layerwise finetuning test since torch is not importable'))
        return

    additional_params = {
        'use_layerwise_tuning': True,
        'batch_size': 20,
        'num_samples_for_tuning': 40,
    }
    algorithm_config = make_algo_config(algo_name, preset, additional_params=additional_params)
    model = models.get(model_name, model_framework, tmp_path)
    reference_name = model_name + '_quantized_tuned'
    metrics, quantized_model = run_algo(model, model_name, algorithm_config, tmp_path, reference_name)
    for metric_name in metrics:
        print('{}: {:.4f}'.format(metric_name, metrics[metric_name]))

    assert metrics == pytest.approx(expected_accuracy, abs=0.006)
    check_model(tmp_path, quantized_model, reference_name,
                model_framework, check_weights=False)

# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from addict import Dict

from openvino.tools.pot.graph import load_model
from openvino.tools.pot.statistics.collector import collect_statistics
from openvino.tools.pot.algorithms.quantization.accuracy_aware.algorithm import AccuracyAwareQuantization
from openvino.tools.pot.engines.ac_engine import ACEngine
from openvino.tools.pot.utils.logger import init_logger
from .utils.config import get_engine_config

init_logger(level='INFO')

TEST_MODELS = [
    ('mobilenet-v2-pytorch', 'pytorch', {'accuracy@top1': 0.999, 'accuracy@top5': 0.999})]


@pytest.mark.parametrize(
    'model_name, model_framework, expected_accuracy', TEST_MODELS,
    ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MODELS])
def test_annotation_free(model_name, model_framework, expected_accuracy, models, tmp_path):
    compression_params = Dict({
        "target_device": "CPU",
        "stat_subset_size": 300,
        "maximal_drop": 1.00,
        "base_algorithm": "MinMaxQuantization",
        "preset": "performance",
        "annotation_free": True,
        "annotation_conf_threshold": 0.6
    })
    model_config = models.get(model_name, model_framework, tmp_path).model_params
    engine_config = get_engine_config(model_name)
    engine_config.models[0].datasets[0].subsample_size = 1000
    metrics = Dict()

    model = load_model(model_config)
    engine = ACEngine(engine_config)
    accuracy_aware_algo = AccuracyAwareQuantization(compression_params, engine)
    collect_statistics(engine, model, [accuracy_aware_algo])
    quantized_model = accuracy_aware_algo.run(model)

    assert accuracy_aware_algo._dataset_size == pytest.approx(721, abs=5)  # pylint: disable=W0212

    engine.set_model(quantized_model)
    metrics.update(engine.predict(print_progress=True)[0])

    for metric, value in metrics.items():
        print('{}: {:.4f}'.format(metric, value))

    assert metrics == pytest.approx(expected_accuracy, abs=0.002)

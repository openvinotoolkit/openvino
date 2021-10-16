# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from addict import Dict

from openvino.tools.pot.engines.creator import create_engine
from openvino.tools.pot.graph import load_model
from openvino.tools.pot.pipeline.initializer import create_pipeline
from .utils.check_graph import check_model
from .utils.config import get_engine_config, merge_configs

TEST_MODELS = [
    # ('resnet-50-pytorch', 'pytorch')
]


@pytest.fixture(scope='module', params=TEST_MODELS,
                ids=['{}_{}'.format(*m) for m in TEST_MODELS])
def _params(request):
    return request.param


def test_ranger_graph(_params, tmp_path, models):
    model_name, model_framework = _params

    algorithm_config = Dict({
        'algorithms': [{
            'name': 'Ranger',
            'params': {
                'target_device': 'ANY',
                'stat_subset_size': 100
            }
        }]
    })

    model = models.get(model_name, model_framework, tmp_path)

    engine_config = get_engine_config(model_name)
    config = merge_configs(model.model_params, engine_config, algorithm_config)

    model = load_model(config.model)
    engine = create_engine(config.engine, data_loader=None, metric=None)
    pipeline = create_pipeline(config.compression.algorithms, engine)

    optimized_model = pipeline.run(model)
    check_model(tmp_path, optimized_model, model_name + '_ranger', model_framework)

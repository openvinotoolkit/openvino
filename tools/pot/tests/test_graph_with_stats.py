# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from addict import Dict
from openvino.tools.pot.data_loaders.creator import create_data_loader
from openvino.tools.pot.engines.creator import create_engine
from openvino.tools.pot.graph import load_model
from openvino.tools.pot.pipeline.initializer import create_pipeline

from .utils.check_graph import check_model
from .utils.config import merge_configs

TEST_MODELS = [
    ('resnet_example', 'pytorch', 'DefaultQuantization', 'CPU'),
    ('resnet_example', 'pytorch', 'RangeSupervision', 'ANY'),
    ('act_act_example', 'pytorch', 'DefaultQuantization', 'GNA')
]


@pytest.fixture(scope='module', params=TEST_MODELS,
                ids=['_'.join(m) for m in TEST_MODELS])
def _params(request):
    return request.param


def test_graph_with_stats(_params, tmp_path, models):
    model_name, model_framework, algo_name, target_device = _params

    algorithm_config = Dict({
        'algorithms': [{
            'name': algo_name,
            'params': {
                'target_device': target_device,
                'preset': 'performance',
                'stat_subset_size': 1
            }
        }]
    })

    model = models.get(model_name, model_framework, tmp_path)

    test_dir = Path(__file__).parent
    path_image_data = os.path.join(test_dir, 'data/image_data')
    engine_config = Dict({'device': 'CPU',
                          'type': 'simplified',
                          'data_source': path_image_data})
    config = merge_configs(model.model_params, engine_config, algorithm_config)

    model = load_model(config.model)
    data_loader = create_data_loader(engine_config, model)
    engine = create_engine(config.engine, data_loader=data_loader, metric=None)
    pipeline = create_pipeline(config.compression.algorithms, engine)

    optimized_model = pipeline.run(model)
    final_model_name = model_name + '_' + algo_name
    check_model(tmp_path, optimized_model, final_model_name, model_framework)

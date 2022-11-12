# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
from addict import Dict

from openvino.tools.pot import create_pipeline, load_model
from openvino.tools.pot.engines.simplified_engine import SimplifiedEngine
from openvino.tools.pot.graph.model_utils import get_nodes_by_type, get_node_by_name
from openvino.tools.pot.graph.node_utils import get_bias_for_node, get_node_value
from openvino.tools.pot.graph.special_operations import OPERATIONS_WITH_BIAS

from .utils.config import merge_configs
from .utils.data_helper import dump_intermediate_data, load_json
from .test_scales import RandomDataLoader


EPS = 1e-6


TEST_TRANSFORMERS_MODELS = [
    (
        'transformer_example',
        'pytorch',
        'DefaultQuantization',
        {
            'preset': 'performance',
            'stat_subset_size': 10,
            'threshold': 1000,
            'target_device': 'CPU'
        },
        'transformer_example_fbc'
    ),
    (
        'transformer_example',
        'pytorch',
        'DefaultQuantization',
        {
            'preset': 'performance',
            'stat_subset_size': 10,
            'threshold': 1000,
            'target_device': 'CPU',
            'use_fast_bias': False
        },
        'transformer_example_hbc'
    )
]


@pytest.fixture(scope='module', params=TEST_TRANSFORMERS_MODELS,
                ids=['{}_{}_{}'.format(*m) for m in TEST_TRANSFORMERS_MODELS])
def _params(request):
    return request.param


def test_transformer_biases_after_correction(_params, tmp_path, models):
    model_name, framework, algorithm, algorithm_params, test_name = _params

    references_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  './data/reference_biases')
    reference_path = os.path.join(references_dir, f'{test_name}.json')
    reference_exists = os.path.isfile(reference_path)
    local_path = os.path.join(tmp_path, f'{test_name}.json')

    model = models.get(model_name, framework, tmp_path)
    algorithm_config = Dict({
        'algorithms': [{
            'name': algorithm,
            'params': algorithm_params
        }]
    })
    engine_config = {'device': 'CPU'}
    config = merge_configs(model.model_params, engine_config, algorithm_config)

    data_loader = RandomDataLoader(shapes={'input': (1, 128)}, seed=0)

    engine = SimplifiedEngine(config.engine, data_loader=data_loader)
    pipeline = create_pipeline(config.compression.algorithms, engine)
    model = load_model(config.model)
    compressed_model = pipeline.run(model)

    values = {}

    nodes_with_bias = get_nodes_by_type(
        compressed_model, [op['type'] for op in OPERATIONS_WITH_BIAS])
    for node_with_bias in nodes_with_bias:
        bias = get_bias_for_node(node_with_bias)
        if bias is not None:
            values[node_with_bias.fullname] = get_node_value(bias)

    if not reference_exists:
        dump_intermediate_data(reference_path, values)
        return
    dump_intermediate_data(local_path, values)

    references = load_json(reference_path)

    assert len(references) == len(values)
    for node_name, ref_val in references.items():
        node = get_node_by_name(compressed_model, node_name)
        bias = get_bias_for_node(node)
        cur_val = get_node_value(bias)
        np.testing.assert_almost_equal(cur_val, ref_val)

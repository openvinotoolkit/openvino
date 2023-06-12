# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np

import pytest
from addict import Dict

from openvino.tools.pot.graph import load_model
from openvino.tools.pot.graph.model_utils import get_node_by_name
from openvino.tools.pot.graph import node_utils as nu
from openvino.tools.pot.algorithms.quantization import fake_quantize as fqut
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.engines.ac_engine import ACEngine
from .utils.config import get_engine_config, merge_configs
from .utils.path import TEST_ROOT
from .utils.data_helper import load_json

TEST_MODELS = [
    ('concat_depthwise_model', 'pytorch', 'MinMaxQuantization', 'accuracy', 'CPU'),
]

REFERENCES_PATH = TEST_ROOT / 'data/reference_unify'


@pytest.fixture(scope='module', params=TEST_MODELS,
                ids=["{}_{}_{}_{}".format(*m) for m in TEST_MODELS])
def _params(request):
    return request.param


def test_unify_scales(_params, tmp_path, models):
    model_name, model_framework, algorithm, preset, device = _params

    if model_framework == 'mxnet':
        pytest.skip('Skipped due to conflict with numpy version in mxnet #99501.')

    algorithm_config = Dict({
        'algorithms': [{
            'name': algorithm,
            'params': {
                'target_device': device,
                'preset': preset,
                'stat_subset_size': 2
            }
        }]
    })

    def _test_unify_scales(model_, to_unify_):
        for _, fqs in to_unify_:
            ranges = []
            for fq in fqs:
                fq = get_node_by_name(model_, fq)
                fq_inputs = nu.get_node_inputs(fq)[1:]
                ranges.append(tuple(fqut.get_node_value(fq_input) for fq_input in fq_inputs))
                assert all([np.array_equal(r, ranges[0][i]) for i, r in enumerate(ranges[-1])])

    model = models.get(model_name, model_framework, tmp_path)

    engine_config = get_engine_config(model_name)
    config = merge_configs(model.model_params, engine_config, algorithm_config)

    model = load_model(config.model)
    pipeline = create_pipeline(config.compression.algorithms, ACEngine(config.engine))
    compressed_model = pipeline.run(model)

    to_unify = fqut.find_fqs_to_unify(compressed_model, config.compression.algorithms[0]['params'])
    _test_unify_scales(compressed_model, to_unify)

    ref_path = REFERENCES_PATH.joinpath(model_name + '_to_unify.json')
    if ref_path.exists():
        to_unify_ref = load_json(ref_path.as_posix())
        assert to_unify == to_unify_ref
    else:
        with open(ref_path.as_posix(), 'w+') as f:
            json.dump(to_unify, f, indent=4)

# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from addict import Dict

import pytest

from openvino.tools.pot import OutlierChannelSplitting
from openvino.tools.pot.graph import load_model
from tests.utils.check_graph import check_model

TEST_WEIGHTS_EXPANSION_RATIO = [0.1, 0.5]
TEST_MODEL_NAME = 'outlier_channel_splitting_example'
TEST_MODEL_FRAMEWORK = 'onnx'


@pytest.mark.parametrize('weights_expansion_ratio', TEST_WEIGHTS_EXPANSION_RATIO,
                         ids=['weights_expansion_ratio_{}'.format(ratio) for ratio in TEST_WEIGHTS_EXPANSION_RATIO])
def test_outlier_channel_splitting_algo(models, tmp_path, weights_expansion_ratio):
    algorithm_config = Dict({
        'weights_expansion_ratio': weights_expansion_ratio,
    })

    model = models.get(TEST_MODEL_NAME, TEST_MODEL_FRAMEWORK, tmp_path)
    model = load_model(model.model_params)

    algorithm = OutlierChannelSplitting(algorithm_config, None)
    algorithm.run(model)

    check_model(tmp_path, model, TEST_MODEL_NAME + '_{}'.format(weights_expansion_ratio),
                TEST_MODEL_FRAMEWORK, check_weights=True)

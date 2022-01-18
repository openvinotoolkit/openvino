# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
from addict import Dict

from openvino.tools.pot.graph import load_model
from openvino.tools.pot.utils.ac_imports import ConfigReader
from openvino.tools.pot.engines.ac_engine import ACEngine
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.configs.hardware_config import HardwareConfig
from openvino.tools.pot.configs.config import Config
from openvino.tools.pot.algorithms.quantization.fake_quantize_configuration import get_configurations_by_preset
from .utils.config import provide_dataset_path
from .utils.path import HARDWARE_CONFIG_PATH, HARDWARE_CONFIG_REFERENCE_PATH, \
    TOOL_CONFIG_PATH, INTERMEDIATE_CONFIG_PATH


def check_hardware_config(config, config_name):
    path_to_ref_json = HARDWARE_CONFIG_REFERENCE_PATH.joinpath(config_name.split('.')[0] + '_flatten.json')
    if not path_to_ref_json.exists():
        with open(path_to_ref_json.as_posix(), 'w') as f:
            json.dump(config, f)

    with open(path_to_ref_json.as_posix(), 'r') as f:
        ref_config = json.load(f)

    assert config == ref_config


HW_CONFIG_NAME = ['cpu.json',
                  'template.json']
TOOL_CONFIG_NAME = ['mobilenet-v2-pytorch_single_dataset.json',
                    'mobilenet-v2-pytorch_divided_datasets.json',
                    'mobilenet-v2-pytorch_range_estimator.json']
TEST_MODEL = ('mobilenet-v2-pytorch', 'pytorch')
PRESETS = [
    'performance',
    'mixed',
    'accuracy'
]


@pytest.mark.parametrize(
    'hw_config_name', HW_CONFIG_NAME,
    ids=['{}_config'.format(os.path.splitext(c)[0]) for c in HW_CONFIG_NAME]
)
def test_load_hardware_config(hw_config_name):
    hw_config_path = HARDWARE_CONFIG_PATH.joinpath(hw_config_name).as_posix()
    hw_config = HardwareConfig.from_json(hw_config_path)
    check_hardware_config(hw_config, hw_config_name)


@pytest.mark.parametrize(
    'config_name', TOOL_CONFIG_NAME,
    ids=['{}_config'.format(os.path.splitext(c)[0]) for c in TOOL_CONFIG_NAME]
)
def test_load_tool_config(config_name, tmp_path, models):
    tool_config_path = TOOL_CONFIG_PATH.joinpath(config_name).as_posix()
    config = Config.read_config(tool_config_path)
    config.configure_params()

    config.engine.log_dir = tmp_path.as_posix()
    config.engine.evaluate = True

    model_name, model_framework = TEST_MODEL
    model = models.get(model_name, model_framework, tmp_path)
    config.model.model = model.model_params.model
    config.model.weights = model.model_params.weights
    provide_dataset_path(config.engine)
    ConfigReader.convert_paths(config.engine)

    pipeline = create_pipeline(config.compression.algorithms, ACEngine(config.engine))

    model = load_model(config.model)
    assert not isinstance(model, int)
    assert pipeline.run(model)


@pytest.mark.parametrize(
    'preset', PRESETS,
    ids=['{}'.format(m) for m in PRESETS]
)
def test_configurations_by_preset(preset):
    def _load_config(name):
        path_to_conf = INTERMEDIATE_CONFIG_PATH.joinpath(name).as_posix()
        with open(path_to_conf, 'r') as f:
            return json.load(f)

    config = Dict({
        'preset': preset,
        'target_device': 'CPU'
    })
    correct_configuration = _load_config('correct_configuration.json')
    res = get_configurations_by_preset(config, None, correct_configuration)
    ref_configuration = _load_config('ref_configuration.json')
    assert res == ref_configuration[preset]

# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import json
from unittest.mock import patch
import pytest
try:
    from openvino_telemetry import Telemetry
    func = 'openvino_telemetry.Telemetry.send_event'
except ImportError:
    try:
        from openvino.tools.pot.utils.telemetry_stub import Telemetry
        func = 'openvino.tools.pot.utils.telemetry_stub.Telemetry.send_event'
    except ImportError:
        pass

from openvino.tools.pot.graph import load_model
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.engines.ac_engine import ACEngine
from openvino.tools.pot.utils.ac_imports import ConfigReader
from openvino.tools.pot.configs.config import Config
from .utils.config import provide_dataset_path
from .utils.path import TELEMETRY_CONFIG_PATH

TEST_MODEL = ('mobilenet-v2-pytorch', 'pytorch')
TOOL_CONFIG_NAME = ['mobilenet-v2-pytorch.json', 'mobilenet-v2-pytorch_aa.json', 'mobilenet-v2-pytorch_sparsity.json']


# pylint: disable=W0221
class TelemetryTest(Telemetry):
    def __init__(self):
        super().__init__(app_name='pot', app_version=None, tid=None)
        self.value = set()

    def send_event(self, event_category, event_label, event_value):
        data = "('{}', '{}', '{}')".format(event_category, event_label, event_value)
        self.value.add(data)


@pytest.mark.parametrize(
    'config_name', TOOL_CONFIG_NAME,
    ids=['{}'.format(os.path.splitext(c)[0]) for c in TOOL_CONFIG_NAME]
)
def test_telemetry(config_name, tmp_path, models):
    telemetry = TelemetryTest()
    with open(os.path.join(TELEMETRY_CONFIG_PATH, 'expected_values.txt')) as file:
        expected = json.load(file)

    @patch(func, new=telemetry.send_event)
    def compress_model():
        telemetry.value = set()
        tool_config_path = TELEMETRY_CONFIG_PATH.joinpath(config_name).as_posix()
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

        pipeline = create_pipeline(config.compression.algorithms, ACEngine(config.engine), 'CLI')
        model = load_model(config.model)
        pipeline.run(model)

        assert set(telemetry.value) == set(expected[config_name])
    compress_model()

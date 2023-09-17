# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.configs.config import Config
from .utils.path import TOOL_CONFIG_PATH

DEVICE = [
    'CPU',
    'GPU',
    'NPU'
]


def test_target_device():
    def read_config(filename):
        tool_config_path = TOOL_CONFIG_PATH.joinpath(filename).as_posix()
        config = Config.read_config(tool_config_path)
        config.configure_params()
        return config['compression']['algorithms'][0]['params']['target_device']

    target_device = read_config('mobilenet-v2-pytorch_single_dataset_without_target_device.json')
    assert target_device == 'ANY'

    target_device = read_config('mobilenet-v2-pytorch_single_dataset.json')
    assert target_device in DEVICE

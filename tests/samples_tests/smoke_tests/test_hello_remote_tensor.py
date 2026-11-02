# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import re
import subprocess

import pytest

from common.samples_common_test_class import download, get_cmd_output, get_devices

MODEL = 'bvlcalexnet-12.onnx'
IMAGE = 'dog-224x224.bmp'

# The sample runs on directly selected GPU devices only; take them from TEST_DEVICE like the other sample tests.
GPU_DEVICES = [device for device in get_devices() if device.startswith('GPU')]


def get_executable():
    return pathlib.Path(os.environ['IE_APP_PATH'], 'hello_remote_tensor').with_suffix('.exe' if os.name == 'nt' else '')


@pytest.fixture
def executable():
    path = get_executable()
    if not path.is_file():
        pytest.skip('The optional Level Zero remote tensor sample was not built')
    return path


@pytest.mark.parametrize('arguments, message', [
    (['model.xml', 'image.bmp', 'CPU'], 'Select a GPU device directly'),
    (['model.xml', 'image.bmp'], 'Usage'),
])
def test_hello_remote_tensor_invalid_arguments(executable, arguments, message):
    result = subprocess.run([executable, *arguments], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert message in result.stdout + result.stderr, result.stdout + result.stderr


@pytest.mark.skipif(not GPU_DEVICES, reason='No GPU device in TEST_DEVICE')
@pytest.mark.parametrize('device', GPU_DEVICES)
def test_hello_remote_tensor_classification(executable, device, cache):
    test_data_dir = cache.mkdir('test_data')
    model = download(test_data_dir, test_data_dir / MODEL)
    image = download(test_data_dir, test_data_dir / IMAGE)

    output = get_cmd_output(executable, model, image, device)

    top1 = None
    for line in output.split('\n'):
        line = line.replace('[ INFO ]', '').strip()
        if re.match(r'\d+ +\d+.\d+$', line):
            top1 = line.split(' ')[0]
            break
    assert top1 == '215', f'Expected top1 class 215, got {top1}\n{output}'

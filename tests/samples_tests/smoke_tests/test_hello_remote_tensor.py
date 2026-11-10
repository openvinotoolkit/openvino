# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import re
import shutil
import subprocess
import sys

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
        message = f'The optional Level Zero remote tensor sample was not built: {path}'
        # A job that must have the sample sets this flag so a failed build is not hidden by a skip.
        if os.environ.get('OV_REQUIRE_REMOTE_TENSOR_SAMPLE') == '1':
            pytest.fail(message)
        pytest.skip(message)
    return path


def test_hello_remote_tensor_missing_arguments(executable):
    result = subprocess.run([executable, 'model.xml', 'image.bmp'], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert 'Usage' in result.stdout + result.stderr, result.stdout + result.stderr


def test_hello_remote_tensor_non_gpu_device(executable):
    # OpenVINO itself rejects devices without a default remote context.
    result = subprocess.run([executable, 'model.xml', 'image.bmp', 'CPU'], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert (result.stdout + result.stderr).strip()


def get_top1(output):
    for line in output.split('\n'):
        line = line.replace('[ INFO ]', '').strip()
        if re.match(r'\d+ +\d+.\d+$', line):
            return line.split(' ')[0]
    return None


@pytest.mark.skipif(not GPU_DEVICES, reason='No GPU device in TEST_DEVICE')
@pytest.mark.parametrize('device', GPU_DEVICES)
def test_hello_remote_tensor_classification(executable, device, cache):
    test_data_dir = cache.mkdir('test_data')
    model = download(test_data_dir, test_data_dir / MODEL)
    image = download(test_data_dir, test_data_dir / IMAGE)

    output = get_cmd_output(executable, model, image, device)

    top1 = get_top1(output)
    assert top1 == '215', f'Expected top1 class 215, got {top1}\n{output}'


# Russian, Hebrew, Czech, Chinese, Korean, Japanese and all of them combined.
UNICODE_NAMES = ['русский', 'עברית', 'česky', '中国人', '한국인', '日本人',
                 'русский_עברית_česky_中国人_한국인_日本人']


@pytest.mark.skipif(not GPU_DEVICES, reason='No GPU device in TEST_DEVICE')
@pytest.mark.skipif(sys.platform.startswith('win'),
                    reason='The samples\' wstring2string helper truncates wide characters, '
                           'so Unicode paths do not work on Windows (see test_hello_classification)')
@pytest.mark.parametrize('name', UNICODE_NAMES)
def test_hello_remote_tensor_unicode_paths(executable, cache, tmp_path, name):
    test_data_dir = cache.mkdir('test_data')
    model = download(test_data_dir, test_data_dir / MODEL)
    image = download(test_data_dir, test_data_dir / IMAGE)

    unicode_dir = tmp_path / name
    unicode_dir.mkdir()
    unicode_model = unicode_dir / (name + pathlib.Path(model).suffix)
    unicode_image = unicode_dir / (name + pathlib.Path(image).suffix)
    shutil.copy(model, unicode_model)
    shutil.copy(image, unicode_image)

    output = get_cmd_output(executable, unicode_model, unicode_image, GPU_DEVICES[0])

    top1 = get_top1(output)
    assert top1 == '215', f'Expected top1 class 215, got {top1}\n{output}'

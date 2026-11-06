# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from openvino import Layout, PartialShape
from openvino.tools.benchmark.utils.inputs_filling import parse_path
from openvino.tools.benchmark.utils.utils import AppInputInfo


def input_info(name, shape, layout):
    info = AppInputInfo()
    info.name, info.node_name = name, name
    info.layout, info.partial_shape = Layout(layout), PartialShape(shape)
    return info


@pytest.mark.parametrize('extension', ['bin', 'npy'])
def test_nc_input_is_not_skipped_without_an_image(tmp_path, extension):
    path = tmp_path / f'input.{extension}'
    path.write_bytes(b'input')
    info = input_info('features', [2, 3], 'NC')
    images, arrays, binaries = parse_path(str(path), [info])
    assert not images
    assert dict(arrays if extension == 'npy' else binaries) == {'features': [str(path)]}


def test_image_info_does_not_reduce_number_of_images_used(tmp_path):
    paths = [tmp_path / f'{i}.png' for i in range(3)]
    for path in paths:
        path.write_bytes(b'image')
    image = input_info('image', [1, 3, 4, 5], 'NCHW')
    metadata = input_info('info', [1, 3], 'NC')
    images, arrays, binaries = parse_path(','.join(str(path) for path in paths), [image, metadata])
    assert dict(images) == {'image': [str(path) for path in paths]}
    assert not arrays
    assert not binaries


def test_multiple_nc_inputs_receive_files_in_order(tmp_path):
    paths = [tmp_path / f'{i}.bin' for i in range(2)]
    for path in paths:
        path.write_bytes(b'input')
    inputs = [input_info(name, [1, 3], 'NC') for name in ['left', 'right']]
    _, _, binaries = parse_path(','.join(str(path) for path in paths), inputs)
    assert dict(binaries) == {'left': [str(paths[0])], 'right': [str(paths[1])]}

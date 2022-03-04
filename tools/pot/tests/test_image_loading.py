# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from addict import Dict

import pytest

from openvino.tools.pot.data_loaders.utils import collect_img_files
from openvino.tools.pot.data_loaders.creator import create_data_loader
from openvino.tools.pot.graph import load_model


TEST_MODELS = [('mobilenet-v2-pytorch', 'pytorch')]


def test_image_loading():
    test_dir = Path(__file__).parent
    image_files = collect_img_files(str(test_dir / 'data/image_loading/image_files.txt'))

    assert len(image_files) == 5
    for i, file_name in enumerate(image_files):
        assert os.path.basename(file_name) == '{}.JPEG'.format(i)


@pytest.mark.parametrize(
    'model_name, model_framework', TEST_MODELS,
    ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MODELS])
def test_check_image(tmp_path, models, model_name, model_framework):
    test_dir = Path(__file__).parent
    path_image_data = os.path.join(test_dir, "data/image_data")

    engine_config = Dict({"device": "CPU",
                          "type": "simplified",
                          "data_source": path_image_data})
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)

    data_loader = create_data_loader(engine_config, model)

    num_images_from_data_loader = len(list(data_loader))

    num_images_in_dir = len(os.listdir(path_image_data))

    assert num_images_from_data_loader == num_images_in_dir


TEST_MODELS_LAYOUT = [
    ('mobilenet-v2-pytorch', 'pytorch', 'NCHW', (3, 224, 224)),
    ('mobilenet-v1-1.0-224-tf', 'tf', 'NHWC', (224, 224, 3)),
    ('mobilenet-v2-pytorch', 'pytorch', None, (3, 224, 224)),
    ('mobilenet-v1-1.0-224-tf', 'tf', None, (224, 224, 3))
]


@pytest.mark.parametrize(
    'model_name, model_framework, layout, reference_shape', TEST_MODELS_LAYOUT,
    ids=['{}_{}_{}_{}'.format(m[0], m[1], m[2], m[3]) for m in TEST_MODELS_LAYOUT])
def test_check_layout(tmp_path, models, model_name, model_framework, layout, reference_shape):
    test_dir = Path(__file__).parent
    path_image_data = os.path.join(test_dir, "data/image_data")

    engine_config = Dict({"device": "CPU",
                          "type": "simplified",
                          "layout": layout,
                          "data_source": path_image_data})
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)

    data_loader = create_data_loader(engine_config, model)
    image = next(iter(data_loader))

    assert image.shape == reference_shape

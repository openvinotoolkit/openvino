# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from addict import Dict

import pytest

from openvino.tools.pot.data_loaders.creator import create_data_loader
from openvino.tools.pot.graph import load_model
from openvino.tools.pot.graph.model_utils import get_nodes_by_type


TEST_MODELS = [
    ('mobilenet-v2-pytorch', 'pytorch', None, None),
    ('mobilenet-v2-pytorch', 'pytorch', None, (3, 640, 720)),
    ('mobilenet-v2-pytorch', 'pytorch', 'HWC', (224, 224, 3)),
    ('mobilenet-v2-pytorch', 'pytorch', 'NHWC', (1, 224, 224, 3)),
    ('mobilenet-v2-pytorch', 'pytorch', 'CHW', (3, 224, 224)),
    ('mobilenet-v2-pytorch', 'pytorch', 'NCHW', (1, 3, 224, 224)),
]

@pytest.mark.parametrize(
    'model_name, model_framework, layout, input_shape', TEST_MODELS,
    ids=['{}_{}_{}_{}'.format(m[0], m[1], m[2], m[3]) for m in TEST_MODELS])
def test_generate_image(tmp_path, models, model_name, model_framework, layout, input_shape):
    path_image_data = os.path.join(tmp_path, 'pot_dataset')
    stat_subset_size = 5
    engine_config = Dict({'device': 'CPU',
                          'type': 'data_free',
                          'data_source': path_image_data,
                          'subset_size': stat_subset_size,
                          'layout': layout,
                          'shape': input_shape,
                          'generate_data': 'True'})
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)
    data_loader = create_data_loader(engine_config, model)

    num_images_from_data_loader = len(list(data_loader))
    num_images_in_dir = len(os.listdir(path_image_data))
    assert num_images_from_data_loader == num_images_in_dir == stat_subset_size

    image = data_loader[0]
    if input_shape is None:
        in_node = get_nodes_by_type(model, ['Parameter'], recursively=False)[0]
        input_shape = tuple(in_node.shape[1:])
    elif len(input_shape) == 4:
        input_shape = input_shape[1:]

    assert image.shape == input_shape

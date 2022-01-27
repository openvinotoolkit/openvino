# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from addict import Dict

from openvino.tools.pot.algorithms.quantization.fake_quantize_configuration import \
    read_all_fake_quantize_configurations, get_configurations_by_preset
from openvino.tools.pot.configs.hardware_config import HardwareConfig
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.graph.transformer import GraphTransformer
from .utils.path import HARDWARE_CONFIG_PATH

CONFIGS_PATH = [
    HARDWARE_CONFIG_PATH / 'cpu.json'
]

TEST_MODELS = [('mobilenet-v2-pytorch', 'pytorch')]

ALGORITHM_CONFIG = Dict({
    'name': 'MinMaxQuantization',
    'params': {
        'target_device': 'CPU',
        'preset': 'performance',
        'stat_subset_size': 1,
    }
})


@pytest.mark.parametrize(
    'model_name, model_framework', TEST_MODELS,
    ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MODELS])
@pytest.mark.parametrize(
    'hardware_config_path', CONFIGS_PATH,
    ids=['{}_config'.format(c.stem) for c in CONFIGS_PATH])
def test_per_channel_activations_for_depthwise(tmp_path, models, model_name, model_framework, hardware_config_path):
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)
    hardware_config = HardwareConfig.from_json(hardware_config_path.as_posix())
    model = GraphTransformer(hardware_config).insert_fake_quantize(model)
    fq_configurations = read_all_fake_quantize_configurations(
        ALGORITHM_CONFIG, hardware_config, model)
    ALGORITHM_CONFIG.preset = ALGORITHM_CONFIG.params.preset
    ALGORITHM_CONFIG.target_device = ALGORITHM_CONFIG.params.target_device
    fq_configuration = get_configurations_by_preset(ALGORITHM_CONFIG, model, fq_configurations, hardware_config)
    fq_dw_names = ['Conv_4/WithoutBiases/fq_input_0', 'Conv_13/WithoutBiases/fq_input_0',
                   'Conv_22/WithoutBiases/fq_input_0', 'Conv_32/WithoutBiases/fq_input_0',
                   'Conv_41/WithoutBiases/fq_input_0', 'Conv_51/WithoutBiases/fq_input_0',
                   'Conv_61/WithoutBiases/fq_input_0', 'Conv_70/WithoutBiases/fq_input_0',
                   'Conv_80/WithoutBiases/fq_input_0', 'Conv_90/WithoutBiases/fq_input_0',
                   'Conv_100/WithoutBiases/fq_input_0', 'Conv_109/WithoutBiases/fq_input_0',
                   'Conv_119/WithoutBiases/fq_input_0', 'Conv_129/WithoutBiases/fq_input_0',
                   'Conv_138/WithoutBiases/fq_input_0', 'Conv_148/WithoutBiases/fq_input_0',
                   'Conv_158/WithoutBiases/fq_input_0']
    dw_config = None
    for config_by_type in hardware_config:
        if config_by_type['type'] == 'DepthWiseConvolution':
            dw_config = config_by_type['quantization']['activations'][0]

    if not dw_config:
        raise Exception('DepthWise missing at hardware configuration')

    save_model(model, tmp_path.as_posix(), model_name)

    for fq_name in fq_configuration:
        if fq_name in fq_dw_names:
            fq_config = fq_configuration[fq_name]['activations']
            assert fq_config == dw_config

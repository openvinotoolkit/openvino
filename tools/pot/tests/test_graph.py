# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from addict import Dict

import pytest

import openvino.tools.pot.graph.node_utils as nu
from openvino.tools.pot.graph import load_model
from openvino.tools.pot.configs.hardware_config import HardwareConfig
from openvino.tools.pot.graph.transformer import GraphTransformer
from openvino.tools.pot.graph.model_utils import get_nodes_by_type, get_node_by_name
from openvino.tools.pot.graph.node_utils import get_node_inputs, get_first_convolutions
from tests.utils.path import HARDWARE_CONFIG_PATH
from tests.utils.check_graph import check_model

CPU_CONFIG_PATH = HARDWARE_CONFIG_PATH / 'cpu.json'
GNA_CONFIG_PATH = HARDWARE_CONFIG_PATH / 'gna.json'

TEST_MODELS = [
    ('mobilenetv2_example', 'pytorch', 'ANY'),
    ('resnet_example', 'pytorch', 'ANY'),
    ('googlenet_example', 'pytorch', 'ANY'),
    ('mobilenetv2_ssd_example', 'pytorch', 'ANY'),
    ('densenet121_example', 'pytorch', 'ANY'),
    ('multiple_out_ports_net', 'tf', 'ANY'),
    ('gru_example', 'pytorch', 'GNA'),
    ('lstm_example', 'pytorch', 'GNA'),
    #('multiple_outputs_net_example', 'tf', 'GNA'),
    ('resnet_example', 'pytorch', 'CPU_SPR'),
    #('tensor_iterator_example', 'tf', 'ANY'),
    ('softsign_example', 'tf', 'GNA'),
]


CASCADE_MAP = Dict({
    'mtcnn': {
        'model_names': ['mtcnn-p', 'mtcnn-r', 'mtcnn-o'],
        'model_tokens': ['pnet', 'rnet', 'onet'],
        'main_model': 'mtcnn-o'
    }
})


@pytest.mark.parametrize(
    'model_name, model_framework, target_device', TEST_MODELS,
    ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MODELS])
def test_build_quantization_graph(tmp_path, models, model_name, model_framework, target_device):
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params, target_device=target_device)

    if target_device == 'GNA':
        hardware_config = HardwareConfig.from_json(GNA_CONFIG_PATH.as_posix())
    else:
        hardware_config = HardwareConfig.from_json(CPU_CONFIG_PATH.as_posix())

    quantization_model = GraphTransformer(hardware_config).insert_fake_quantize(model)

    check_model(tmp_path, quantization_model, model_name, model_framework)


MODELS_FOR_TESTING_IGNORED_PARAMS = [
    ('mobilenetv2_example', 'pytorch'),
    ('resnet_example', 'pytorch'),
    ('googlenet_example', 'pytorch'),
    ('mtcnn', 'caffe')
]


@pytest.mark.parametrize(
    'model_name, model_framework', MODELS_FOR_TESTING_IGNORED_PARAMS,
    ids=['{}_{}'.format(m[0], m[1]) for m in MODELS_FOR_TESTING_IGNORED_PARAMS])
def test_build_quantization_graph_with_ignored_params(
        tmp_path, models, model_name, model_framework):
    if model_name in CASCADE_MAP:
        model = models.get_cascade(model_name, model_framework, tmp_path, CASCADE_MAP[model_name])
    else:
        model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)
    hardware_config = HardwareConfig.from_json(CPU_CONFIG_PATH.as_posix())

    if model_name not in CASCADE_MAP:
        ignored_params = {
            'operations': [
                {
                    'type': 'Add',
                },
                {
                    'type': 'Convolution',
                    'attributes': {
                        'output': 1280
                    }
                }
            ]
        }

    if model_name == 'resnet_example':
        ignored_params['scope'] = ['Conv_11/WithoutBiases', 'Conv_29/WithoutBiases']
    elif model_name == 'googlenet_example':
        node_name = 'Conv_10/WithoutBiases'
        ignored_params['scope'] = [node_name]
    elif model_name == 'mtcnn':
        ignored_params = {
            'pnet': {
                'scope': ['conv1/WithoutBiases', 'conv3/WithoutBiases']
            },
            'rnet': {
                'skip_model': True
            },
            'onet': {
                'operations': [
                    {
                        'type': 'MatMul'
                    }
                ]
            }
        }

    quantization_model = GraphTransformer(hardware_config).insert_fake_quantize(model, ignored_params)

    print(len(get_nodes_by_type(quantization_model, ['FakeQuantize'])))
    check_model(tmp_path, quantization_model, model_name + '_ig_params', model_framework)


@pytest.mark.parametrize(
    'model_name, model_framework', MODELS_FOR_TESTING_IGNORED_PARAMS,
    ids=['{}_{}'.format(m[0], m[1]) for m in MODELS_FOR_TESTING_IGNORED_PARAMS])
def test_build_quantization_graph_with_ignored_agnostic_params(
        tmp_path, models, model_name, model_framework):
    if model_name in CASCADE_MAP:
        model = models.get_cascade(model_name, model_framework, tmp_path, CASCADE_MAP[model_name])
    else:
        model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)
    hardware_config = HardwareConfig.from_json(CPU_CONFIG_PATH.as_posix())
    if model_name not in CASCADE_MAP:
        ignored_params = {
            'scope': [],
            'operations': [{'type': 'MaxPool'},
                           {'type': 'Reshape'}]
        }

    if model_name == 'mtcnn':
        ignored_params = {
            'pnet': {'scope': [], 'operations': [{'type': 'MaxPool'}]},
            'rnet': {'skip_model': True, 'scope': [], 'operations': [{'type': 'MaxPool'}]},
            'onet': {'scope': [], 'operations': [{'type': 'MaxPool'}]}
        }

    quantization_model = GraphTransformer(hardware_config).insert_fake_quantize(model, ignored_params)

    for model_dict in quantization_model.models:
        model = model_dict['model']
        dict_ignored_operation_model = ignored_params[model_dict['name']]['operations'] \
            if quantization_model.is_cascade else ignored_params['operations']
        ignored_params_operation = [op['type'] for op in dict_ignored_operation_model]
        for node in model.get_op_nodes():
            if node.type in ignored_params_operation:
                parent_type = [str(n.type) for n in nu.get_node_inputs(node) if n is not None]
                assert 'FakeQuantize' not in parent_type


TEST_MODELS_REMOVAL = [
    ('mobilenetv2_ssd_example', 'pytorch', ['Conv_12/WithoutBiases',
                                            'Conv_26/WithoutBiases',
                                            'Conv_41/WithoutBiases']),
    ('squeezenet1_1_example', 'pytorch', ['Conv_5/WithoutBiases',
                                          'Conv_47/WithoutBiases']),
    ('mobilenetv2_example', 'pytorch', ['Conv_10/WithoutBiases',
                                        'Conv_18/WithoutBiases',
                                        'Conv_60/WithoutBiases']),
    ('googlenet_example', 'pytorch', ['Conv_5/WithoutBiases',
                                      'Conv_10/WithoutBiases',
                                      'Conv_19/WithoutBiases',
                                      'Conv_87/WithoutBiases',
                                      'Conv_93/WithoutBiases']),
    ('multiple_out_ports_net', 'tf', ['add_indices'])
]


def cut_fq_node(model, node_list, graph_transformer, tmp_path):
    model_ = load_model(model.model_params)
    quantized_model = graph_transformer.insert_fake_quantize(model_)
    cropped_model = quantized_model
    for node_name in node_list:
        node = get_node_by_name(cropped_model, node_name)
        for parent_node in nu.get_node_inputs(node):
            if parent_node and parent_node and parent_node.type == 'FakeQuantize':
                cropped_model, *_ = graph_transformer.remove_fq_nodes(quantized_model, [parent_node.name])
                break

    check_model(tmp_path, cropped_model, model.model_name + '_cut_fq', model.framework)


@pytest.mark.parametrize(
    'model_name, model_framework, node_list', TEST_MODELS_REMOVAL,
    ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MODELS_REMOVAL])
def test_cutting_fq_layers(tmp_path, models, model_name, model_framework, node_list):
    model = models.get(model_name, model_framework, tmp_path)
    hardware_config = HardwareConfig.from_json(CPU_CONFIG_PATH.as_posix())
    graph_transformer = GraphTransformer(hardware_config)

    cut_fq_node(model, node_list, graph_transformer, tmp_path)


TEST_MODELS_WITH_PATTERNS = [
    ('efficientnet_b0_example', 'pytorch'),
    ('mobilenet_v3_small_example', 'pytorch'),
    # ('image-retrieval-0001', 'dldt'),
    ('scaleshift_fuse', 'pytorch'),
    ('scaleshift_no_fuse_1', 'pytorch'),
    ('scaleshift_no_fuse_2', 'pytorch'),
    ('matmul_divide_const', 'tf')
]


@pytest.mark.parametrize(
    'model_name, model_framework', TEST_MODELS_WITH_PATTERNS,
    ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MODELS_WITH_PATTERNS])
def test_build_quantization_graph_with_ignored_blocks(tmp_path, models, model_name, model_framework):
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)
    hardware_config = HardwareConfig.from_json(CPU_CONFIG_PATH.as_posix())
    quantization_model = GraphTransformer(hardware_config).insert_fake_quantize(model)

    check_model(tmp_path, quantization_model, model_name + '_ig_pt', model_framework)


TEST_MODELS_WITHOUT_FQ_MOVING = [
    ('test_multibranch_propogation_without_fq_moving', 'pytorch')
]


@pytest.mark.parametrize(
    'model_name, model_framework', TEST_MODELS_WITHOUT_FQ_MOVING,
    ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MODELS_WITHOUT_FQ_MOVING])
def test_multibranch_propagation_without_fq_moving(tmp_path, models, model_name, model_framework):
    ignored_params = {
        # Ignoring quantization for the first 4 convolution in the model
        "scope": ['8/WithoutBiases', '9/WithoutBiases', '10/WithoutBiases', '11/WithoutBiases']
    }

    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)

    hardware_config = HardwareConfig.from_json((HARDWARE_CONFIG_PATH / 'cpu.json').as_posix())
    quantized_model = GraphTransformer(hardware_config).insert_fake_quantize(model, ignored_params)

    # Checking last convolution has FQ at inputs
    node = get_node_by_name(quantized_model, '13/WithoutBiases')
    for node_input in get_node_inputs(node)[:2]:
        assert node_input.type == 'FakeQuantize'
    # Checking ignored convolutions has no quantizers on inputs
    assert len(get_nodes_by_type(quantized_model, ['FakeQuantize'])) == 2


MODELS_WITH_LSTM = [
    ('lstm_example', 'pytorch', {
        'LSTM_15/TensorIterator/22/variable_1':
            ['Assign_304'],
        'LSTM_15/TensorIterator/24/variable_2':
            ['Assign_311'],
        'LSTM_19/TensorIterator/22/variable_1':
            ['Assign_333'],
        'LSTM_19/TensorIterator/24/variable_2':
            ['Assign_340']
    })
]


def test_lstm_ends(tmp_path, models):
    model_name, model_framework, lstm_ends_ref = MODELS_WITH_LSTM[0]
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)
    read_values = get_nodes_by_type(model, ['ReadValue'])
    assigns = get_nodes_by_type(model, ['Assign'])
    for read_value in read_values:
        assert read_value.name in lstm_ends_ref
        lstm_ends = nu.get_lstm_ends(read_value, assigns, [])
        lstm_ends_names = [n.name for n in lstm_ends]
        assert sorted(lstm_ends_names) == sorted(lstm_ends_ref[read_value.name])


TEST_MODELS_WITHOUT_FQ_MOVING = [
    ('test_multibranch_propogation_with_fq_moving', 'pytorch')
]


@pytest.mark.parametrize(
    'model_name, model_framework', TEST_MODELS_WITHOUT_FQ_MOVING,
    ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MODELS_WITHOUT_FQ_MOVING])
def test_multibranch_propagation_with_fq_moving(tmp_path, models, model_name, model_framework):
    ignored_params = {
        # Ignoring quantization for the first 4 convolution in the model
        "scope": ['8/WithoutBiases', '9/WithoutBiases', '10/WithoutBiases', '11/WithoutBiases']
    }

    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)

    hardware_config = HardwareConfig.from_json((HARDWARE_CONFIG_PATH / 'cpu.json').as_posix())
    quantized_model = GraphTransformer(hardware_config).insert_fake_quantize(model, ignored_params)

    node = get_node_by_name(quantized_model, '14')
    for node_input in get_node_inputs(node)[:2]:
        assert node_input.type == 'FakeQuantize'
    assert get_node_inputs(node)[2].type == 'Concat'

    node = get_node_by_name(quantized_model, '12')
    for node_input in get_node_inputs(node)[:2]:
        assert node_input.type == 'FakeQuantize'

    assert len(get_nodes_by_type(quantized_model, ['FakeQuantize'])) == 6


MODELS_FOR_FIRST_CONV_TEST = [
    ('1_input_model', 'onnx', ['Conv_3/WithoutBiases']),
    ('3_inputs_model', 'onnx', ['Conv_3/WithoutBiases', 'Conv_5/WithoutBiases', 'Conv_7/WithoutBiases']),
]


@pytest.mark.parametrize(
    'model_name, model_framework, first_convs_ref', MODELS_FOR_FIRST_CONV_TEST,
    ids=['{}_{}'.format(m[0], m[1]) for m in MODELS_FOR_FIRST_CONV_TEST])
def test_first_convolutions_search(tmp_path, models, model_name, model_framework, first_convs_ref):
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)
    input_nodes = get_nodes_by_type(model, ['Parameter'])
    first_convs = get_first_convolutions(input_nodes)
    first_convs_names = [n.name for n in first_convs]
    assert sorted(first_convs_names) == sorted(first_convs_ref)

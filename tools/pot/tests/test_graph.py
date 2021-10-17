# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from addict import Dict

import pytest

import openvino.tools.pot.graph.node_utils as nu
from openvino.tools.pot.graph import load_model
from openvino.tools.pot.configs.hardware_config import HardwareConfig
from openvino.tools.pot.graph.transformer import GraphTransformer
from openvino.tools.pot.graph.model_utils import get_nodes_by_type, get_node_by_name
from openvino.tools.pot.graph.node_utils import get_node_inputs, get_first_convolutions
from tests.utils.path import TEST_ROOT, HARDWARE_CONFIG_PATH
from tests.utils.check_graph import check_model

CPU_CONFIG_PATH = HARDWARE_CONFIG_PATH / 'cpu.json'
GNA_CONFIG_PATH = HARDWARE_CONFIG_PATH / 'gna.json'

TEST_MODELS = [
    # ('mobilenet-v2-pytorch', 'pytorch'),
    # ('resnet-50-pytorch', 'pytorch'),
    # ('googlenet-v3', 'tf'),
    # ('ssd_mobilenet_v2_coco', 'tf'),
    # ('densenet-121', 'caffe'),
    ('multiple_out_ports_net', 'tf'),  # multiple output ports in node case check,
    # ('rm_nnet4a', 'kaldi')
]

CASCADE_MAP = Dict({
    'mtcnn': {
        'model_names': ['mtcnn-p', 'mtcnn-r', 'mtcnn-o'],
        'model_tokens': ['pnet', 'rnet', 'onet'],
        'main_model': 'mtcnn-o'
    }
})


@pytest.mark.parametrize(
    'model_name, model_framework', TEST_MODELS,
    ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MODELS])
def test_build_quantization_graph(tmp_path, models, model_name, model_framework):
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)

    hardware_config = HardwareConfig.from_json(CPU_CONFIG_PATH.as_posix())
    if model_framework == 'kaldi':
        hardware_config = HardwareConfig.from_json(GNA_CONFIG_PATH.as_posix())

    quantization_model = GraphTransformer(hardware_config).insert_fake_quantize(model)

    check_model(tmp_path, quantization_model, model_name, model_framework)


MODELS_FOR_TESTING_IGNORED_PARAMS = [
    # ('mobilenet-v2-pytorch', 'pytorch'),
    # ('resnet-50-pytorch', 'pytorch'),
    # ('googlenet-v3', 'tf'),
    # ('mtcnn', 'caffe')
]


@pytest.mark.parametrize(
    'model_name, model_framework', MODELS_FOR_TESTING_IGNORED_PARAMS[1:],
    ids=['{}_{}'.format(m[0], m[1]) for m in MODELS_FOR_TESTING_IGNORED_PARAMS[1:]])
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
                        'output': 1280,
                        'group': 1
                    }
                }
            ]
        }

    if model_name == 'resnet-50-pytorch':
        ignored_params['scope'] = ['Conv_5/WithoutBiases', 'Conv_18/WithoutBiases']
    elif model_name == 'googlenet-v3':
        node_name = 'InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0a_1x1/convolution'
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
    'model_name, model_framework', MODELS_FOR_TESTING_IGNORED_PARAMS[:],
    ids=['{}_{}'.format(m[0], m[1]) for m in MODELS_FOR_TESTING_IGNORED_PARAMS[:]])
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
    # ('mobilenet-ssd', 'caffe', ['conv1/WithoutBiases',
    #                             'conv16_2_mbox_conf/WithoutBiases',
    #                             'conv15_1/WithoutBiases']),
    # ('ssd_resnet50_512', 'mxnet', ['stage1_unit1_sc',
    #                                'stage3_unit1_sc',
    #                                'multi_feat_2_conv_3x3_relu_cls_pred_conv/WithoutBiases']),
    # ('squeezenet1.1', 'pytorch', ['fire4/expand3x3/WithoutBiases',
    #                               'fire8/expand3x3/WithoutBiases']),
    # ('mobilenet-v2-pytorch', 'pytorch', ['Conv_18/WithoutBiases',
    #                                      'Conv_66/WithoutBiases',
    #                                      'Conv_162/WithoutBiases']),
    # ('googlenet-v3', 'tf', ['InceptionV3/InceptionV3/Conv2d_3b_1x1/convolution',
    #                         'InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_0a_1x1/convolution',
    #                         'InceptionV3/InceptionV3/Mixed_6d/Branch_0/Conv2d_0a_1x1/convolution',
    #                         'InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0a_1x1/convolution',
    #                         'InceptionV3/InceptionV3/Mixed_6a/Branch_0/Conv2d_1a_1x1/convolution']),
    ('multiple_out_ports_net', 'tf', ['add_indices'])
]


@pytest.fixture(scope='module', params=TEST_MODELS_REMOVAL,
                ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MODELS_REMOVAL])
def _params(request):
    return request.param


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


def test_cutting_fq_layers(_params, tmp_path, models):
    model_name, model_framework, node_list = _params
    model = models.get(model_name, model_framework, tmp_path)
    hardware_config = HardwareConfig.from_json(CPU_CONFIG_PATH.as_posix())
    graph_transformer = GraphTransformer(hardware_config)

    cut_fq_node(model, node_list, graph_transformer, tmp_path)


TEST_MODELS_WITH_PATTERNS = [
    # ('efficientnet-b0', 'tf'),
    # ('se-resnet-50', 'caffe'),
    # ('image-retrieval-0001', 'dldt'),
    ('scaleshift_fuse', 'dldt'),
    ('scaleshift_no_fuse_1', 'dldt'),
    ('scaleshift_no_fuse_2', 'dldt'),
    ('matmul_divide_const', 'dldt')
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


def test_multibranch_propagation_without_fq_moving():
    TEST_CASES_PATH = TEST_ROOT / 'data' / 'test_cases_refs'
    model_path = (TEST_CASES_PATH / 'test_ig_border_case_without_fq_moving.xml').as_posix()
    weights_path = (TEST_CASES_PATH / 'test_ig_border_case_without_fq_moving.bin').as_posix()

    ignored_params = {
        "scope": ['8/WithoutBiases', '9/WithoutBiases', '10/WithoutBiases', '11/WithoutBiases']
    }

    config = Dict({'model': model_path, 'weights': weights_path})
    model = load_model(config)

    hardware_config = HardwareConfig.from_json((HARDWARE_CONFIG_PATH / 'cpu.json').as_posix())
    quantized_model = GraphTransformer(hardware_config).insert_fake_quantize(model, ignored_params)

    node = get_node_by_name(quantized_model, '13/WithoutBiases')
    for node_input in get_node_inputs(node)[:2]:
        assert node_input.type == 'FakeQuantize'
    assert len(get_nodes_by_type(quantized_model, ['FakeQuantize'])) == 2


MODELS_WITH_LSTM = [
    # ('rm_lstm4f', 'kaldi', {
    #     'prev_memory_output69':
    #         ['next_lstm_output108', 'lstmprojectedstreams/Shape', 'input_fullyconnected/WithoutBiases'],
    #     'prev_memory_state82':
    #         ['state_filtered_tahn100', 'clamp_scaleshift101/Mul_', 'next_lstm_state98'],
    #     'prev_memory_output':
    #         ['next_lstm_output', 'affinetransform/WithoutBiases'],
    #     'prev_memory_state':
    #         ['state_filtered_tahn', 'clamp_scaleshift/Mul_', 'next_lstm_state']
    # })
]


@pytest.fixture(scope='module', params=MODELS_WITH_LSTM,
                ids=['{}_{}'.format(m[0], m[1]) for m in MODELS_WITH_LSTM])
def _params(request):
    return request.param


def test_lstm_ends(_params, tmp_path, models):
    model_name, model_framework, lstm_ends_ref = _params
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)
    read_values = get_nodes_by_type(model, ['ReadValue'])
    assigns = get_nodes_by_type(model, ['Assign'])
    for read_value in read_values:
        assert read_value.name in lstm_ends_ref
        lstm_ends = nu.get_lstm_ends(read_value, assigns, [])
        lstm_ends_names = [n.name for n in lstm_ends]
        assert sorted(lstm_ends_names) == sorted(lstm_ends_ref[read_value.name])


def test_multibranch_propagation_with_fq_moving():
    TEST_CASES_PATH = TEST_ROOT / 'data' / 'test_cases_refs'
    model_path = (TEST_CASES_PATH / 'test_ig_border_case_with_fq_moving.xml').as_posix()
    weights_path = (TEST_CASES_PATH / 'test_ig_border_case_with_fq_moving.bin').as_posix()

    ignored_params = {
        "scope": ['8/WithoutBiases', '9/WithoutBiases', '10/WithoutBiases', '11/WithoutBiases']
    }

    config = Dict({'model': model_path, 'weights': weights_path})
    model = load_model(config)

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


@pytest.fixture(scope='module', params=MODELS_FOR_FIRST_CONV_TEST,
                ids=['{}_{}'.format(m[0], m[1]) for m in MODELS_FOR_FIRST_CONV_TEST])
def _params(request):
    return request.param


def test_first_convolutions_search(_params, tmp_path, models):
    model_name, model_framework, first_convs_ref = _params
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)
    input_nodes = get_nodes_by_type(model, ['Parameter'])
    first_convs = get_first_convolutions(input_nodes)
    first_convs_names = [n.name for n in first_convs]
    assert sorted(first_convs_names) == sorted(first_convs_ref)

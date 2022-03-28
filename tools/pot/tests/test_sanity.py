# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from copy import deepcopy
from pathlib import Path

from collections import OrderedDict
import json

import pytest
from addict import Dict

import cv2 as cv
import numpy as np

from openvino.tools.pot.app.run import optimize
from openvino.tools.pot.graph import save_model, load_model
from tools.evaluate import evaluate
from .utils.check_graph import check_model
from .utils.config import get_engine_config, merge_configs, \
    get_dataset_info, PATHS2DATASETS_CONFIG, make_algo_config

TEST_MODELS = [
    ('mobilenet-v2-pytorch', 'pytorch', 'DefaultQuantization', 'performance', 300, {'accuracy@top1': 0.737,
                                                                                    'accuracy@top5': 0.909},
     {}, 'CPU'),

    ('mobilenet-v2-pytorch', 'pytorch', 'DefaultQuantization', 'mixed', 300, {'accuracy@top1': 0.731,
                                                                              'accuracy@top5': 0.908},
     {}, 'CPU'),

    ('mobilenet-v1-1.0-224-tf', 'tf', 'DefaultQuantization', 'performance', 100, {'accuracy@top1': 0.728,
                                                                                  'accuracy@top5': 0.909},
     {'use_fast_bias': False}, 'CPU'),

    ('mobilenet-v1-1.0-224-tf', 'tf', 'DefaultQuantization', 'performance', 100, {'accuracy@top1': 0.728,
                                                                                  'accuracy@top5': 0.911},
     {}, 'CPU'),

    ('mobilenet-ssd', 'caffe', 'AccuracyAwareQuantization', 'performance', 300, {'map': 0.6801},
     {'metric_subset_ratio': 1.0, 'max_iter_num': 1, 'metrics': [{'name': 'map', 'baseline_value': 0.669}]}, 'CPU'),

    ('mobilenet-ssd', 'caffe', 'AccuracyAwareQuantization', 'performance', 300, {'map': 0.6801},
     {'metric_subset_ratio': 1.0, 'max_iter_num': 1, 'tune_hyperparams': True,
      'metrics': [{'name': 'map', 'baseline_value': 0.669}]}, 'CPU'),

    # ('mobilenet-v1-0.25-128', 'tf', 'AccuracyAwareQuantization', 'performance', 100,
    # {'accuracy@top1': 0.424, 'accuracy@top5': 0.65},
    # {'drop_type': 'relative', 'max_iter_num': 1, 'accuracy_drop': 0.005, 'metrics': [
    #     {'name': 'accuracy@top1', 'baseline_value': 0.431}]}, 'GNA'),

    ('mtcnn', 'caffe', 'DefaultQuantization', 'performance', 1, {'recall': 0.76, 'map': 0.6618}, {}, 'CPU'),

    ('mtcnn', 'caffe', 'DefaultQuantization', 'performance', 2, {'recall': 0.68, 'map': 0.4406},
     {'use_fast_bias': False}, 'CPU'),
    ('octave-resnet-26-0.25', 'mxnet', 'DefaultQuantization', 'performance', 300,
     {'accuracy@top1': 0.766, 'accuracy@top5': 0.927}, {'use_fast_bias': False}, 'CPU'),
]
CASCADE_MAP = Dict({
    'mtcnn': {
        'model_names': ['mtcnn-p', 'mtcnn-r', 'mtcnn-o'],
        'model_tokens': ['pnet', 'rnet', 'onet'],
        'main_model': 'mtcnn-o'
    }
})


@pytest.fixture(scope='module', params=TEST_MODELS,
                ids=['{}_{}_{}_{}'.format(*m) for m in TEST_MODELS])
def _params(request):
    return request.param


def test_compression(_params, tmp_path, models):
    model_name, model_framework, algorithm, preset, subset_size, expected_accuracy, additional_params, device = _params

    algorithm_config = make_algo_config(algorithm, preset, subset_size, additional_params, device)

    if model_name in CASCADE_MAP:
        model = models.get_cascade(model_name, model_framework, tmp_path, CASCADE_MAP[model_name])
    else:
        model = models.get(model_name, model_framework, tmp_path)

    engine_config = get_engine_config(model_name)
    config = merge_configs(model.model_params, engine_config, algorithm_config)
    if model_name in CASCADE_MAP:
        config.engine.evaluations[0].module_config.datasets[0].subsample_size = 10
    else:
        config.engine.models[0].datasets[0].subsample_size = 1000

    metrics = optimize(config)

    output_dir = os.path.join(config.model.exec_log_dir, 'optimized')

    for metric_name in metrics:
        print('{}: {:.4f}'.format(metric_name, metrics[metric_name]))

    assert metrics == pytest.approx(expected_accuracy, abs=0.006)
    if model_name in CASCADE_MAP:
        for token in CASCADE_MAP.model_name.model_tokens:
            assert os.path.exists(os.path.join(output_dir, '{}_{}.xml'.format(config.model.model_name, token)))
            assert os.path.exists(os.path.join(output_dir, '{}_{}.bin'.format(config.model.model_name, token)))
    else:
        assert os.path.exists(os.path.join(output_dir, config.model.model_name + '.xml'))
        assert os.path.exists(os.path.join(output_dir, config.model.model_name + '.bin'))

    if device == 'GNA' and algorithm == 'AccuracyAwareQuantization':
        quantized_model_params = deepcopy(model.model_params)
        quantized_model_params['model'] = os.path.join(output_dir, config.model.model_name + '.xml')
        quantized_model_params['weights'] = os.path.join(output_dir, config.model.model_name + '.bin')
        quantized_model = load_model(quantized_model_params)
        check_model(tmp_path, quantized_model, model_name + '_gna_aa', model_framework)


TEST_SAMPLE_MODELS = [
    ('mobilenet-v2-1.0-224', 'tf', 'DefaultQuantization', 'performance', {'accuracy@top1': 0.716}, []),
    ('mobilenet-v2-1.0-224', 'tf', 'DefaultQuantization', 'performance', {'accuracy@top1': 0.716},
     ['--input_shape=[1,?,?,3]'])
]


@pytest.fixture(scope='module', params=TEST_SAMPLE_MODELS,
                ids=['{}_{}_{}_{}'.format(*m) for m in TEST_SAMPLE_MODELS])
def _sample_params(request):
    return request.param


def test_sample_compression(_sample_params, tmp_path, models):
    model_name, model_framework, algorithm, preset, expected_accuracy, custom_mo_config = _sample_params

    # hack for sample imports because sample app works only from sample directory
    pot_dir = Path(__file__).parent.parent
    sys.path.append(str(pot_dir / 'sample'))
    # pylint: disable=C0415
    from openvino.tools.pot.api.samples.classification.classification_sample import optimize_model

    model = models.get(model_name, model_framework, tmp_path, custom_mo_config=custom_mo_config)
    data_source, annotations = get_dataset_info('imagenet_1001_classes')

    args = Dict({
        'model': model.model_params.model,
        'dataset': data_source,
        'annotation_file': annotations['annotation_file']})

    model_, _ = optimize_model(args)

    paths = save_model(model_, tmp_path.as_posix(), model_name)
    model_xml = os.path.join(tmp_path.as_posix(), '{}.xml'.format(model_name))
    weights = os.path.join(tmp_path.as_posix(), '{}.bin'.format(model_name))

    assert os.path.exists(model_xml)
    assert os.path.exists(weights)

    algorithm_config = make_algo_config(algorithm, preset)
    engine_config = get_engine_config(model_name)
    config = merge_configs(model.model_params, engine_config, algorithm_config)
    config.engine = get_engine_config(model_name)

    metrics = evaluate(config=config, subset=range(1000), paths=paths)

    metrics = OrderedDict([(metric.name, np.mean(metric.evaluated_value))
                           for metric in metrics])

    for metric_name, metric_val in metrics.items():
        print('{}: {:.4f}'.format(metric_name, metric_val))
        if metric_name == 'accuracy@top1':
            assert {metric_name: metric_val} == pytest.approx(expected_accuracy, abs=0.006)


SIMPLIFIED_TEST_MODELS = [
    ('mobilenet-v2-pytorch', 'pytorch', 'DefaultQuantization', 'performance',
     {'accuracy@top1': 0.701, 'accuracy@top5': 0.91}, []),
    ('mobilenet-v2-pytorch', 'pytorch', 'DefaultQuantization', 'performance',
     {'accuracy@top1': 0.712, 'accuracy@top5': 0.906}, ['--input_shape=[1,3,?,?]'])
]


def launch_simplified_mode(_simplified_params, tmp_path, models, engine_config):
    model_name, model_framework, algorithm, preset, _, custom_mo_config = _simplified_params
    algorithm_config = make_algo_config(algorithm, preset)

    model = models.get(model_name, model_framework, tmp_path, custom_mo_config=custom_mo_config)
    config = merge_configs(model.model_params, engine_config, algorithm_config)

    _ = optimize(config)

    output_dir = os.path.join(config.model.exec_log_dir, 'optimized')
    model = os.path.join(output_dir, config.model.model_name + '.xml')
    weights = os.path.join(output_dir, config.model.model_name + '.bin')

    assert os.path.exists(model)
    assert os.path.exists(weights)

    paths = [{
        'model': model,
        'weights': weights
    }]

    config.engine = get_engine_config(model_name)
    metrics = evaluate(
        config=config, subset=range(1000), paths=paths)

    metrics = OrderedDict([(metric.name, np.mean(metric.evaluated_value))
                           for metric in metrics])

    for metric_name, metric_val in metrics.items():
        print('{}: {:.4f}'.format(metric_name, metric_val))

    return metrics


@pytest.fixture(scope='module', params=SIMPLIFIED_TEST_MODELS,
                ids=['{}_{}_{}_{}'.format(*m) for m in SIMPLIFIED_TEST_MODELS])
def _simplified_params(request):
    return request.param

def test_simplified_mode(_simplified_params, tmp_path, models):
    with open(PATHS2DATASETS_CONFIG.as_posix()) as f:
        data_source = Dict(json.load(f))['ImageNet2012'].pop('source_dir')

    engine_config = Dict({'type': 'simplified',
                          'data_source': '{}/{}'.format(data_source, 'ILSVRC2012_val*'),
                          'device': 'CPU',
                          'central_fraction': 0.875})

    _, _, _, _, expected_accuracy, _ = _simplified_params
    metrics = launch_simplified_mode(_simplified_params, tmp_path, models, engine_config)
    assert metrics == pytest.approx(expected_accuracy, abs=0.006)


def test_frame_extractor_tool():
    # hack due to strange python imports (same as in sample test)
    pot_dir = Path(__file__).parent.parent
    sys.path.append(str(pot_dir / 'tools/frame_extractor'))
    # pylint: disable=C0415
    from tools.frame_extractor.extractor import extract_frames_and_make_dataset

    test_dir = Path(__file__).parent
    test_video_path = test_dir / 'data/video/video_example.avi'
    output_dir = test_dir / 'data/frame_extractor'
    dataset_size, frame_step = 3, 1

    extract_frames_and_make_dataset(
        test_video_path.as_posix(),
        output_dir, dataset_size, frame_step, ext='png')

    out_dir = str(test_dir / 'data/frame_extractor')
    ref_dir = str(test_dir / 'data/frame_extractor_ref')

    for i in range(3):
        test = cv.imread('{}/{}.png'.format(out_dir, i))
        ref = cv.imread('{}/{}.png'.format(ref_dir, i))

        assert np.linalg.norm(test - ref) < 1e-3


TEST_MULTIPLE_OUT_PORTS = [('multiple_out_ports_net', 'tf')]


@pytest.mark.parametrize(
    'model_name, model_framework', TEST_MULTIPLE_OUT_PORTS,
    ids=['{}_{}'.format(m[0], m[1]) for m in TEST_MULTIPLE_OUT_PORTS])
def test_multiport_outputs_model(tmp_path, models, model_name, model_framework):
    test_dir = Path(__file__).parent
    # one image as dataset
    data_source = (test_dir / 'data/image_data/').as_posix()
    engine_config = Dict({'type': 'simplified',
                          'data_source': data_source,
                          'device': 'CPU'})

    model = models.get(model_name, model_framework, tmp_path)
    algorithm_config = make_algo_config('MinMaxQuantization', 'performance')
    config = merge_configs(model.model_params, engine_config, algorithm_config)

    _ = optimize(config)

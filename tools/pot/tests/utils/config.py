# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from addict import Dict
try:
    import jstyleson as json
except ImportError:
    import json

from openvino.tools.pot.utils.ac_imports import ConfigReader
from openvino.tools.pot.configs.config import Config
from ..utils.path import ENGINE_CONFIG_PATH, DATASET_CONFIG_PATH
from .open_model_zoo import download_engine_config


PATHS2DATASETS_CONFIG = DATASET_CONFIG_PATH/'dataset_path.json'
NAME_FROM_DEFINITIONS2DATASET_NAME = {
    'imagenet_1000_classes': 'ImageNet2012',
    'imagenet_1000_classes_2015': 'ImageNet2012',
    'imagenet_1001_classes': 'ImageNet2012',
    'VOC2012': 'VOC2007',
    'VOC2007': 'VOC2007',
    'VOC2007_detection': 'VOC2007',
    'wider': 'WiderFace',
    'ms_coco_mask_rcnn': 'COCO2017',
    'ms_coco_detection_91_classes': 'COCO2017',
    'VOC2012_Segmentation': 'VOC2012_Segmentation',
    'ms_coco_detection_80_class_without_background': 'COCO2017'
}


def find_engine_config_locally(model_name):
    configs = Dict()
    for root, _, files in os.walk(ENGINE_CONFIG_PATH.as_posix()):
        for file in files:
            if file.endswith('.json'):
                configs[file.rstrip('.json')] = os.path.join(root, file)
    if model_name in configs.keys():
        with open(configs[model_name]) as f:
            engine_config = Dict(json.load(f))
        return engine_config
    return None


def get_engine_config(model_name):
    engine_config = find_engine_config_locally(model_name)
    if not engine_config:
        engine_config = download_engine_config(model_name)
        if not engine_config:
            raise FileNotFoundError

    mode = 'evaluations' if engine_config.module else 'models'
    engine_config = Dict({mode: [engine_config]})
    if not model_name == 'ncf':
        provide_dataset_path(engine_config)

    sub_root = engine_config[mode][0] if mode == 'models' \
        else engine_config[mode][0].module_config

    sub_root.launchers[0].device = 'CPU'

    if sub_root.datasets[0].annotation:
        sub_root.datasets[0].pop('annotation')

    if sub_root.datasets[0].dataset_meta:
        sub_root.datasets[0].pop('dataset_meta')

    engine_config.evaluate = True
    engine_config.type = 'accuracy_checker'
    ConfigReader.convert_paths(engine_config)

    return engine_config


def get_dataset_info(dataset_name):
    with open(PATHS2DATASETS_CONFIG.as_posix()) as f:
        datasets_paths = Dict(json.load(f))
        paths_config = datasets_paths[NAME_FROM_DEFINITIONS2DATASET_NAME[dataset_name]]
        data_source, dataset_info = paths_config.pop('source_dir'), {}
        for arg, path in paths_config.items():
            if path:
                dataset_info[arg] = path
        return data_source, dataset_info


def provide_dataset_path(config):
    dataset_configs = config.models[0].datasets if config.models \
        else config.evaluations[0].module_config.datasets
    if isinstance(dataset_configs, dict):
        dataset_configs = list(dataset_configs.values())
    for dataset in dataset_configs:
        dataset.data_source, dataset_meta = get_dataset_info(dataset.name)
        if dataset_meta:
            for key, value in dataset_meta.items():
                dataset.annotation_conversion[key] = value


def merge_configs(model_conf, engine_conf, algo_conf):
    config = Config()

    # mo config
    config.model = model_conf

    # ac config
    config.engine = engine_conf

    # algo config
    opt_config = algo_conf.pop('optimizer', None)
    if opt_config is not None:
        config.optimizer = opt_config

    config.compression = algo_conf
    config.add_log_dir(config.model.output_dir, config.model.output_dir)

    return config


def make_algo_config(algorithm, preset, subset_size=300, additional_params=None, device='CPU'):
    params = Dict({
        'target_device': device,
        'preset': preset,
        'stat_subset_size': subset_size
    })
    if additional_params is not None:
        for param_name, param_value in additional_params.items():
            params[param_name] = param_value

    return Dict({
        'algorithms': [{
            'name': algorithm,
            'params': params
        }]
    })

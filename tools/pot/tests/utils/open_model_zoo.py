# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import importlib

try:
    import jstyleson as json
except ImportError:
    import json
from pathlib import Path
from addict import Dict

import openvino
from openvino.tools.pot.utils.ac_imports import ConfigReader

from .command import Command
from .path import LIBS_ROOT, MO_PATH, ENGINE_CONFIG_PATH


TMP_PATH = Path(tempfile.gettempdir())
DOWNLOAD_PATH = TMP_PATH/'open_model_zoo'
CACHE_PATH = TMP_PATH/'open_model_zoo_cache'
OMZ_DOWNLOADER_PATH = LIBS_ROOT/'open_model_zoo'/'tools'/'model_tools'
OMZ_DEFINITIONS_PATH = LIBS_ROOT/'open_model_zoo'/'data'/'dataset_definitions.yml'

sys.path.append(str(OMZ_DOWNLOADER_PATH / 'src'))
# pylint: disable=E0611,C0413,C0411,E0401
importlib.reload(openvino)
from openvino.model_zoo._configuration import load_models, ModelLoadingMode
from openvino.model_zoo._common import MODEL_ROOT
is_platform_windows = sys.platform.startswith('win')


def command_line_for_download(args):
    python_path = OMZ_DOWNLOADER_PATH.as_posix()
    executable = OMZ_DOWNLOADER_PATH.joinpath('downloader.py').as_posix()
    cli_args = ' '.join(key if val is None else '{} {}'.format(key, val) for key, val in args.items())
    cli_args += ' --cache_dir ' + CACHE_PATH.as_posix()
    cli_args += ' --output_dir ' + DOWNLOAD_PATH.as_posix()
    cli_args += ' --num_attempts=5'
    script_launch_cli = '{python_exe} {main_py} {args}'.format(
        python_exe=sys.executable, main_py=executable, args=cli_args
    )
    if not is_platform_windows:
        return 'PYTHONPATH={path} '.format(path=python_path) + script_launch_cli
    return 'cmd /C "set PYTHONPATH={path} && {script_launch_cli}"'.format(
        path=python_path,
        script_launch_cli=script_launch_cli,
    )


def download(config):
    names = config.name
    args = {'--name': ','.join(names if isinstance(names, list) else [names])}
    runner = Command(command_line_for_download(args))
    return runner.run()


def command_line_for_convert(config, custom_mo_config=None):
    python_path = DOWNLOAD_PATH.as_posix()
    executable = OMZ_DOWNLOADER_PATH.joinpath('converter.py').as_posix()
    cli_args = ' -o ' + config.model_params.output_dir
    cli_args += ' -d ' + python_path
    cli_args += ' --name ' + config.name
    cli_args += ' --mo ' + MO_PATH.joinpath('mo.py').as_posix()
    cli_args += ' --precisions ' + config.precision
    if custom_mo_config:
        for custom_mo_arg in custom_mo_config:
            cli_args += ' --add_mo_arg=' + custom_mo_arg
    script_launch_cli = '{python_exe} {main_py} {args}'.format(
        python_exe=sys.executable, main_py=executable, args=cli_args
    )
    if not is_platform_windows:
        return 'PYTHONPATH={path}:$PYTHONPATH '.format(path=python_path) + script_launch_cli
    return 'cmd /C "set PYTHONPATH={path};%PYTHONPATH% && {script_launch_cli}"'.format(
        path=python_path,
        script_launch_cli=script_launch_cli,
    )


def convert(config, custom_mo_config=None):
    runner = Command(command_line_for_convert(config, custom_mo_config))
    return runner.run()


def get_models_list():
    return load_models(MODEL_ROOT, Dict(config=None), mode=ModelLoadingMode.ignore_composite)


def download_engine_config(model_name):
    def process_config():
        engine_conf = Dict()
        mode = 'evaluations' if ac_conf.evaluations else 'models'
        for model in ac_conf[mode]:
            model_ = model
            engine_conf_ = engine_conf
            if mode == 'evaluations':
                engine_conf.module = model.module
                model_ = model.module_config
                engine_conf_ = engine_conf.module_config
                engine_conf_.network_info = model_.network_info

            for launcher in model_.launchers:
                if launcher.framework == 'openvino':
                    engine_conf_.launchers = list()
                    engine_launcher = {'framework': launcher.framework}
                    if launcher.adapter:
                        engine_launcher.update({'adapter': launcher.adapter})
                    engine_conf_.launchers.append(engine_launcher)
                    engine_conf_.datasets = model_.datasets
                    convert_path_to_str(engine_conf)
                    return engine_conf
        return None

    def convert_path_to_str(config):
        iterator = config.items() if isinstance(config, dict) else enumerate(config)
        for key, value in iterator:
            if isinstance(value, (dict, list)):
                convert_path_to_str(value)
            elif isinstance(value, Path):
                config[key] = value.as_posix()

    config_path = LIBS_ROOT / Path('open_model_zoo/models/public/{}/accuracy-check.yml'.format(model_name))
    try:
        ac_conf = Dict(ConfigReader.merge(Dict({
            'config': config_path.as_posix(), 'definitions': OMZ_DEFINITIONS_PATH
        }))[0])
    except FileNotFoundError:
        return None

    engine_config = process_config()
    with open((ENGINE_CONFIG_PATH/Path(model_name+'.json')).as_posix(), 'w') as f:
        json.dump(engine_config, f, indent=4)

    return engine_config

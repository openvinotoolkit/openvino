# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os.path
import string
import sys

from addict import Dict

from .command import Command
from .path import MO_PATH

try:
    import jstyleson as json
except ImportError:
    import json

from .path import TEST_ROOT

CUSTOM_MODELS_PATH = TEST_ROOT/'data'/'models'

is_platform_windows = sys.platform.startswith('win')


def get_models_list():
    models = []
    for root, _, files in os.walk(CUSTOM_MODELS_PATH.as_posix()):
        for file in files:
            if file.endswith('.json'):
                model_config = load_model_config(os.path.join(root, file))
                if model_config:
                    models.append(model_config)
    return models


def load_model_config(path):
    with open(path) as f:
        model_config = Dict(json.load(f))
        mo_args = model_config.mo_args
        if not mo_args:
            return None
        for key, value in mo_args.items():
            if isinstance(value, str):
                mo_args[key] = string.Template(value).substitute(model_dir=CUSTOM_MODELS_PATH.as_posix())
            else:
                mo_args[key] = str(value)
        return model_config


def convert_custom_command_line(config):
    python_path = MO_PATH.as_posix()
    executable = MO_PATH.joinpath('mo.py').as_posix()
    cli_args = ' --model_name ' + config.name
    cli_args += ' --output_dir ' + config.model_params.output_dir
    cli_args += ' --compress_to_fp16=' + "False" if config.precision == "FP32" else "True"
    cli_args += ' '.join([' --' + key for key, value in config.mo_args.items() if value == 'True'])
    cli_args += ' '.join(
        [' --' + key + '=' + value.replace(' ', '') for key, value in config.mo_args.items() if value != 'True'])
    script_launch_cli = '{python_exe} {main_py} {args}'.format(
        python_exe=sys.executable, main_py=executable, args=cli_args
    )
    if not is_platform_windows:
        return 'PYTHONPATH={path}:$PYTHONPATH '.format(path=python_path) + script_launch_cli
    return 'cmd /C "set PYTHONPATH={path};%PYTHONPATH% && {script_launch_cli}"'.format(
        path=python_path,
        script_launch_cli=script_launch_cli,
    )


def convert_custom(config):
    runner = Command(convert_custom_command_line(config))
    return runner.run()

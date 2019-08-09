"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import subprocess
from pathlib import Path
from typing import Union
from collections import namedtuple
from ..utils import get_path, format_key

FrameworkParameters = namedtuple('FrameworkParameters', ['name', 'meta'])

def convert_model(topology_name, model=None, weights=None, meta=None,
                  framework=FrameworkParameters('caffe', False), mo_search_paths=None, mo_params=None, mo_flags=None,
                  tf_custom_op_config_dir=None, tf_object_detection_api_config_dir=None):
    """
    Args:
        topology_name: name for converted model files.
        model: path to the topology file.
        weights: path to the weights file.
        meta: path to the meta file
        framework: framework name for original model.
        mo_search_paths: paths where ModelOptimizer may be found. If None only default paths is used.
        mo_params: value parameters for ModelOptimizer execution.
        mo_flags: flags parameters for ModelOptimizer execution.
        tf_custom_op_config_dir: path to Tensor Flow custom operations directory.
        tf_object_detection_api_config_dir: path to Tensor Flow directory with config for object detection API.
    Returns:
        paths to converted to IE IR model and weights.
    """

    mo_params = mo_params or {}
    mo_flags = mo_flags or []

    set_topology_name(mo_params, topology_name)

    model_optimizer_executable = find_mo(mo_search_paths)
    if not model_optimizer_executable:
        raise EnvironmentError(
            'Model optimizer not found. Please set MO_DIR environment variable to model optimizer folder '
            'installation or refer to help for command line options for providing Model optimizer'
        )

    framework_specific_options = {
        FrameworkParameters('caffe', False): {'input_model': weights, 'input_proto': model},
        FrameworkParameters('mxnet', False): {'input_model': weights},
        FrameworkParameters('tf', False): {'input_model': model},
        FrameworkParameters('tf', True): {'input_meta_graph': meta},
        FrameworkParameters('onnx', False): {'input_model': model},
        FrameworkParameters('kaldi', False): {'input_model': model}
    }

    mo_params['framework'] = framework.name
    mo_params.update(framework_specific_options.get(framework, {}))

    set_path_to_custom_operation_configs(mo_params, framework, tf_custom_op_config_dir, model_optimizer_executable)
    set_path_to_object_detection_api_pipeline_config(mo_params, framework, tf_object_detection_api_config_dir)
    args = prepare_args(str(model_optimizer_executable), flag_options=mo_flags, value_options=mo_params)

    code = exec_mo_binary(args)

    if code.returncode != 0:
        raise RuntimeError("Model optimizer conversion failed: ModelOptimizer returned non-zero code")

    model_file, bin_file = find_dlsdk_ir(
        get_path(mo_params.get('output_dir', Path.cwd()), is_directory=True), mo_params['model_name']
    )
    if not bin_file or not model_file:
        raise RuntimeError("Model optimizer finished correctly, but converted model is not found.")

    return model_file, bin_file


def find_dlsdk_ir(search_path: Path, model_name):
    """
    Args:
        search_path: path with IE IR of model.
        model_name: name of the model.
    Returns:
        paths to IE IR of model.
    """

    xml_file = search_path / '{}.xml'.format(model_name)
    bin_file = search_path / '{}.bin'.format(model_name)

    return get_path(xml_file), get_path(bin_file)


def find_mo(search_paths=None) -> Union[Path, None]:
    """
    Args:
        search_paths: paths where ModelOptimizer may be found. If None only default paths is used.
    Returns:
        path to the ModelOptimizer or None if it wasn't found.
    """

    default_mo_path = ('intel', 'openvino', 'deployment_tools', 'model_optimizer')
    default_paths = [Path.home().joinpath(*default_mo_path), Path('/opt').joinpath(*default_mo_path)]

    executable = 'mo.py'
    for path in search_paths or default_paths:
        path = Path(path)
        if not path.is_dir():
            continue

        mo = path / executable
        if not mo.is_file():
            continue

        return mo

    return None


def prepare_args(executable, flag_options=None, value_options=None):
    """
    Args:
        executable: path to the executable.
        flag_options: positional arguments for executable.
        value_options: keyword arguments for executable.
    Returns:
        list with command-line entries.
    """

    result = [sys.executable, executable]

    for flag_option in flag_options or []:
        result.append(str(format_key(flag_option)))

    for key, value in (value_options or {}).items():
        result.append(str(format_key(key)))
        result.append(str(value))

    return result


def exec_mo_binary(args, timeout=None):
    """
    Args:
        args: command-line entries.
        timeout: timeout for execution.
    Returns:
        result of execution.
    """

    return subprocess.run(args, check=False, timeout=timeout)


def set_path_to_custom_operation_configs(mo_params, framework, tf_custom_op_config_dir, mo_path):
    if framework.name != 'tf':
        return mo_params

    config_path = mo_params.get('tensorflow_use_custom_operations_config')
    if not config_path:
        return mo_params

    if tf_custom_op_config_dir:
        tf_custom_op_config_dir = Path(tf_custom_op_config_dir)
    else:
        tf_custom_op_config_dir = Path('/').joinpath(*mo_path.parts[:-1]) / 'extensions' / 'front' / 'tf'

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = tf_custom_op_config_dir / config_path

    mo_params['tensorflow_use_custom_operations_config'] = str(get_path(config_path))

    return mo_params


def set_path_to_object_detection_api_pipeline_config(mo_params, framework, object_detection_api_config_dir=None):
    object_detection_api_config = mo_params.get('tensorflow_object_detection_api_pipeline_config')
    if framework.name != 'tf' or not object_detection_api_config:
        return mo_params
    model_path = mo_params.get('input_model') or mo_params.get('input_meta_graph')

    object_detection_api_config_dir = Path(object_detection_api_config_dir or get_path(model_path).parent)
    config_path = object_detection_api_config_dir / object_detection_api_config
    mo_params['tensorflow_object_detection_api_pipeline_config'] = str(get_path(config_path))

    return mo_params


def set_topology_name(mo_params, topology_name):
    if not mo_params.get('model_name'):
        mo_params['model_name'] = topology_name

    return mo_params

#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import platform
import subprocess

lib_env_key = "PATH" if platform.system() == "Windows" else "LD_LIBRARY_PATH"
if lib_env_key not in os.environ:
    os.environ[lib_env_key] = ""

python_path_key = "PYTHONPATH"
if python_path_key not in os.environ:
    os.environ[python_path_key] = ""

lib_path_orig = os.environ[lib_env_key]
python_path_orig = os.environ[python_path_key]


def setup_env(module="", libs=[]):
    """
    Update os.environ variables with given values.
    :param module: path to python module
    :param libs: list with paths to libraries
    """
    os.environ[python_path_key] = os.pathsep.join([os.environ[python_path_key], module])
    os.environ[lib_env_key] = os.pathsep.join([os.environ[lib_env_key], *libs])


def reset_env():
    """
    Reset os.environ variables to default values
    """
    os.environ[python_path_key] = python_path_orig
    os.environ[lib_env_key] = lib_path_orig


def try_to_import_ie(module="", libs=[], silent=False):
    """
    Check if Inference Engine Python API modules exists and in case of success
    environment will be set with given values.
    :param module: path to python module
    :param libs: list with paths to libraries
    :param silent: hide all output
    """
    path_to_script = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'check_ie_bindings.py')
    # We need to execute python modules checker in subprocess to avoid issue with environment
    # in case if previous import was unsuccessful it can fail further imports even if sys.path
    # will be restored to initial default values.
    # To pass environment to sub-process PATH/LD_LIBRARY_PATH and PYTHONPATH are used from
    # os.environ that is set after setup_env()
    setup_env(module=module, libs=libs)
    cmd_args = [sys.executable, path_to_script, "--path_to_module", "PYTHONPATH" if module == "" else module]
    if silent:
        cmd_args.append("--silent")

    status = subprocess.run(cmd_args, env=os.environ)
    if status.returncode == 0:
        return True
    else:
        reset_env()
        return False


def find_ie_version(silent=False):
    """
    Tries to import Inference Engine Python API bindings. In case of successful import
    PATH/LD_LIBRARY_PATH and PYTHONPATH environment variables will be set
    This variables must be passed to subprocess in order to execute IE python bindings.
    Example:
        if find_ie_version():
            subprocess.run([sys.executable, path_to_script], env=os.environ)

    """
    if try_to_import_ie(silent=silent):
        return True

    python_version = 'python{}.{}'.format(sys.version_info[0], sys.version_info[1])

    script_path = os.path.realpath(os.path.dirname(__file__))

    # Windows
    bindings_paths_windows = [
        # Package
        {
            "module": os.path.join(script_path, '../../../../python/', python_version),
            "libs": [
                os.path.join(script_path, '../../../inference_engine/bin/intel64/Release'),
                os.path.join(script_path, '../../../inference_engine/external/tbb/bin'),
                os.path.join(script_path, '../../../ngraph/lib'),
            ],
        },
        # Local builds
        {
            "module": os.path.join(script_path, '../../../bin/intel64/Release/python_api/', python_version),
            "libs": [
                os.path.join(script_path, '../../../bin/intel64'),
                os.path.join(script_path, '../../../bin/intel64/Release'),
                os.path.join(script_path, '../../../inference-engine/temp/tbb/bin'),
            ]
        },
        {
            "module": os.path.join(script_path, '../../../bin/intel64/Debug/python_api/', python_version),
            "libs": [
                os.path.join(script_path, '../../../bin/intel64'),
                os.path.join(script_path, '../../../bin/intel64/Debug'),
                os.path.join(script_path, '../../../inference-engine/temp/tbb/bin'),
            ]
        },
    ]

    # Linux / Darwin
    bindings_paths_linux = [
        # Package
        {
            "module": os.path.join(script_path, '../../../../python/', python_version),
            "libs": [
                os.path.join(script_path, '../../../inference_engine/lib/intel64'),
                os.path.join(script_path, '../../../inference_engine/external/tbb/lib'),
                os.path.join(script_path, '../../../ngraph/lib'),
            ],
        },
        # Local builds
        {
            "module": os.path.join(script_path, '../../../bin/intel64/Release/lib/python_api/', python_version),
            "libs": [
                os.path.join(script_path, '../../../bin/intel64/Release/lib'),
            ]
        },

        {
            "module": os.path.join(script_path, '../../../bin/intel64/RelWithDebInfo/lib/python_api/', python_version),
            "libs": [
                os.path.join(script_path, '../../../bin/intel64/RelWithDebInfo/lib'),
            ]
        },
        {
            "module": os.path.join(script_path, '../../../bin/intel64/Debug/lib/python_api/', python_version),
            "libs": [
                os.path.join(script_path, '../../../bin/intel64/Debug/lib'),
            ]
        }
    ]

    bindings_paths = bindings_paths_windows if platform.system() == "Windows" else bindings_paths_linux
    for item in bindings_paths:
        module = item['module']
        if not os.path.exists(module):
            continue
        if try_to_import_ie(module=os.path.normpath(module), libs=item['libs'] if 'libs' in item else [], silent=silent):
            return True

    return False


if __name__ == "__main__":
    if not find_ie_version():
        exit(1)

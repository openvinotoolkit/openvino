#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import os
import sys
import platform
import subprocess
from multiprocessing import Queue
from copy import copy

lib_env_key = "PATH" if platform.system() == "Windows" else "LD_LIBRARY_PATH"
python_path_key = "PYTHONPATH"
ov_frontend_path_key = "OV_FRONTEND_PATH"


def setup_env(env, module="", libs=[]):
    """
    Update env variables with given values.
    :param module: path to python module
    :param libs: list with paths to libraries
    """
    env[python_path_key] = os.pathsep.join([module, env[python_path_key]])
    env[lib_env_key] = os.pathsep.join([*libs, env[lib_env_key]])
    env[ov_frontend_path_key] = os.pathsep.join([*libs])


def try_to_import_ie(env: dict, module="", libs=[], silent=False):
    """
    Check if Inference Engine Python API modules exists and in case of success
    environment will be set with given values.
    :param env: dictionary with predefined environment variables
    :param module: path to python module
    :param libs: list with paths to libraries
    :param silent: hide all output
    """
    path_to_script = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'check_ie_bindings.py')
    # We need to execute python modules checker in subprocess to avoid issue with environment
    # in case if previous import was unsuccessful it can fail further imports even if sys.path
    # will be restored to initial default values.
    # To pass environment to sub-process PATH/LD_LIBRARY_PATH and PYTHONPATH are used from
    # env that is set after setup_env()
    env_orig = copy(env)
    setup_env(env, module=module, libs=libs)
    cmd_args = [sys.executable, path_to_script, "--path_to_module", "PYTHONPATH" if module == "" else module]
    if silent:
        cmd_args.append("--silent")

    status = subprocess.run(cmd_args, env=env)
    if status.returncode == 0:
        return True
    else:
        # reset original values
        for k in env_orig:
            env[k] = env_orig[k]
        return False


def find_ie_version(obj, silent=False):
    """
    Tries to import Inference Engine Python API bindings. In case of successful import
    PATH/LD_LIBRARY_PATH and PYTHONPATH environment variables will be set
    This variables must be passed to subprocess in order to execute IE python bindings.
    Example:
        if find_ie_version():
            subprocess.run([sys.executable, path_to_script], env=env)

    """
    env = obj

    # Other types os comparison like isinstance is not working here, so
    # using WA with type(Queue())
    if type(obj) == type(Queue()):
        env = obj.get()

    if lib_env_key not in env:
        env[lib_env_key] = ""
    if python_path_key not in env:
        env[python_path_key] = ""
    if ov_frontend_path_key not in env:
        env[ov_frontend_path_key] = ""

    if try_to_import_ie(env, silent=silent):
        if type(obj) == type(Queue()):
            obj.put(env)
            obj.put(True)
        return True

    python_version = 'python{}.{}'.format(sys.version_info[0], sys.version_info[1])

    script_path = os.path.realpath(os.path.dirname(__file__))

    # Windows
    bindings_paths_windows = [
        # Local builds
        {
            "module": os.path.join(script_path, '../../../../../../bin/intel64/Release/python_api/', python_version),
            "libs": [
                os.path.join(script_path, '../../../../../../bin/intel64'),
                os.path.join(script_path, '../../../../../../bin/intel64/Release'),
                os.path.join(script_path, '../../../../../../temp/tbb/bin'),
            ]
        },
        {
            "module": os.path.join(script_path, '../../../../../../bin/intel64/Debug/python_api/', python_version),
            "libs": [
                os.path.join(script_path, '../../../../../../bin/intel64'),
                os.path.join(script_path, '../../../../../../bin/intel64/Debug'),
                os.path.join(script_path, '../../../../../../temp/tbb/bin'),
            ]
        },
    ]

    # Linux / Darwin
    bindings_paths_linux = [
        # Local builds
        {
            "module": os.path.join(script_path, '../../../../../../bin/intel64/Release/lib/python_api/', python_version),
            "libs": [
                os.path.join(script_path, '../../../../../../bin/intel64/Release/lib'),
            ]
        },

        {
            "module": os.path.join(script_path, '../../../../../../bin/intel64/RelWithDebInfo/lib/python_api/', python_version),
            "libs": [
                os.path.join(script_path, '../../../../../../bin/intel64/RelWithDebInfo/lib'),
            ]
        },
        {
            "module": os.path.join(script_path, '../../../../../../bin/intel64/Debug/lib/python_api/', python_version),
            "libs": [
                os.path.join(script_path, '../../../../../../bin/intel64/Debug/lib'),
            ]
        }
    ]

    bindings_paths = bindings_paths_windows if platform.system() == "Windows" else bindings_paths_linux
    for item in bindings_paths:
        module = item['module']
        if not os.path.exists(module):
            continue
        if try_to_import_ie(env, module=os.path.normpath(module), libs=item['libs'] if 'libs' in item else [], silent=silent):
            if type(obj) == type(Queue()):
                obj.put(env)
                obj.put(True)
            return True

    if type(obj) == type(Queue()):
        obj.put(False)
    return False


if __name__ == "__main__":
    if not find_ie_version(os.environ):
        exit(1)

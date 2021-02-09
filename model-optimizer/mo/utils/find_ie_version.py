#!/usr/bin/env python3

"""
 Copyright (C) 2021 Intel Corporation

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


def try_to_import_ie(module="", libs=[]):
    """
    Check if IE python modules exists and in case of success
    environment will be set with given values.
    :param module: path to python module
    :param libs: list with paths to libraries
    """
    path_to_script = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'check_ie_bindings.py')
    # We need to execute python modules checker in subprocess to avoid issue with environment
    # in case if previous import was unsuccessful it can fail further imports even if sys.path
    # will be restored to initial default values.
    # To pass environment to sub-process PATH/LD_LIBRARY_PATH and PYTHONPATH are used from
    # os.environ that is set after setup_env()
    setup_env(module=module, libs=libs)
    status = subprocess.run([sys.executable, path_to_script], env=os.environ)
    if status.returncode == 0:
        return True
    else:
        reset_env()
        return False


def find_ie_version():
    """
    Tries to import IE python bindings. In case of successful import
    PATH/LD_LIBRARY_PATH and PYTHONPATH environment variables will be set
    This variables must be passed to subprocess in order to execute IE python bindings.
    Example:
        if find_ie_version():
            subprocess.run([sys.executable, path_to_script], env=os.environ)

    """
    # print("[ IMPORT ] Checking default IE Python module")
    if try_to_import_ie():
        return True

    python_version = 'python{}.{}'.format(sys.version_info[0], sys.version_info[1])

    script_path = os.path.realpath(os.path.dirname(__file__))
    bindings_paths = [
        # Windows
        {
            "module": os.path.join(script_path, '../../../../python/', python_version),
            "libs": [
                os.path.join(script_path, '../../../inference_engine/bin/intel64/Release'),
                os.path.join(script_path, '../../../inference_engine/external/tbb/bin'),
                os.path.join(script_path, '../../../ngraph/lib'),
            ]
        },
        # Linux / Darwin
        {
            "module": os.path.join(script_path, '../../../../python/', python_version),
            "libs": [
                os.path.join(script_path, '../../../inference_engine/lib/intel64'),
                os.path.join(script_path, '../../../inference_engine/external/tbb/lib'),
                os.path.join(script_path, '../../../ngraph/lib'),
            ]
        },
        # Local builds
        {
            "module": os.path.join(script_path, '../../../bin/intel64/Release/lib/python_api/', python_version)
        },
        {
            "module": os.path.join(script_path, '../../../bin/intel64/Debug/lib/python_api/', python_version)
        }
    ]

    for item in bindings_paths:
        # print("[ IMPORT ] Trying to find module in {}".format(item['module']))
        if try_to_import_ie(module=item['module'], libs=item['libs'] if 'libs' in item else []):
            return True

    return False


if __name__ == "__main__":
    if not find_ie_version():
        exit(1)
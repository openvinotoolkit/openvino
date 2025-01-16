# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os
import subprocess # nosec
import sys


def log_ie_not_found():
    log.error("Could not find the OpenVINO or Python API.\n"
              "Consider building the OpenVINO and Python APIs"
              " from sources or try to install OpenVINO (TM) Toolkit using pip \npip install openvino")


def log_mo_root_dir_not_found():
    log.error("Could not find the ModelOptimizer root module directory.\n"
              "Consider setting PYTHONPATH to the openvino tools folder (usually openvino/tools/mo)")


def setup_env():
    mo_root_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)

    # Check that MO root directory already set to the PYTHONPATH
    def is_mo_imported():
        try:
            status = subprocess.run([sys.executable, os.path.join(mo_root_path, 'openvino/tools/mo/utils/check_mo_import.py')],
                                    env=os.environ)
            return status.returncode == 0
        except:
            return False

    if not is_mo_imported():
        # If no, we try to set it manually based on relative path
        python_path_key = 'PYTHONPATH'
        if python_path_key not in os.environ:
            os.environ[python_path_key] = mo_root_path
        else:
            os.environ[python_path_key] = os.pathsep.join([os.environ[python_path_key], mo_root_path])

        sys.path.append(mo_root_path)

        if not is_mo_imported():
            log_mo_root_dir_not_found()
            sys.exit(1)

    ie_found = True
    try:
        from openvino.tools.mo.utils.find_ie_version import find_ie_version  # pylint: disable=no-name-in-module
        ie_found = find_ie_version(silent=True)
    except Exception as e:
        log.error(e)
        ie_found = False

    if not ie_found:
        log_ie_not_found()
        sys.exit(1)

    return True


def subprocess_main(framework=None):
    """
        Please keep this file compatible with python2 in order to check user python version.

        This function checks that OpenVINO Python API available and working as expected
        and then in sub-process it executes main_<fw>.py files. Due to some OSs specifics we can't
        just add paths to Python modules and libraries into current env. So to make OpenVINO
        Python API to be available inside MO we need to use subprocess with new env.
    """
    setup_env()

    path_to_main = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                'main_{}.py'.format(framework) if framework else 'main.py')

    # python2 compatible code. Do not remove.
    args = [sys.executable, path_to_main]

    for arg in sys.argv[1:]:
        args.append(arg)
    status = subprocess.run(args, env=os.environ)
    sys.exit(status.returncode)

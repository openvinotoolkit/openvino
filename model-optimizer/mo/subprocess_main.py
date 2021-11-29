# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess

from mo.utils.versions_checker import check_python_version  # pylint: disable=no-name-in-module


def subprocess_main(framework=None):
    """
        Please keep this file compatible with python2 in order to check user python version.

        This function checks that Inference Engine Python API available and working as expected
        and then in sub-process it executes main_<fw>.py files. Due to some OSs specifics we can't
        just add paths to Python modules and libraries into current env. So to make Inference Engine
        Python API to be available inside MO we need to use subprocess with new env.
    """
    ret_code = check_python_version()
    if ret_code:
        sys.exit(ret_code)

    from mo.utils.find_ie_version import find_ie_version
    find_ie_version(silent=True)

    mo_root_path = os.path.join(os.path.dirname(__file__), os.pardir)

    python_path_key = 'PYTHONPATH'
    if python_path_key not in os.environ:
        os.environ[python_path_key] = mo_root_path
    else:
        os.environ[python_path_key] = os.pathsep.join([os.environ[python_path_key], mo_root_path])

    path_to_main = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                'main_{}.py'.format(framework) if framework else 'main.py')
    # python2 compatible code. Do not remove.
    args = [sys.executable, path_to_main]
    for arg in sys.argv[1:]:
        args.append(arg)
    status = subprocess.run(args, env=os.environ)
    sys.exit(status.returncode)
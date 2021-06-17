# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

from mo.subprocess_main import setup_env


def test_frontends():
    setup_env()
#    args = [sys.executable, 'frontend_ngraph_test_main.py']
    args = [sys.executable, '-m', 'pytest',
            'frontend_ngraph_test_main.py', '-s']

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode

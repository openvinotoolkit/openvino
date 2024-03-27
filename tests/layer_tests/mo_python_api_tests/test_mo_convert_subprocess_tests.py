# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import unittest


class TestSubprocessMoConvert(unittest.TestCase):
    def test_mo_convert(self):
        args = [sys.executable, '-m', 'pytest',
                os.path.join(os.path.dirname(__file__), 'mo_convert_legacy_extensions_test_actual.py'), '-s']

        status = subprocess.run(args, env=os.environ)
        assert not status.returncode
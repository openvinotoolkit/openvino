# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import unittest
from unittest.mock import patch

from mo.main import main
from mo.utils.error import FrameworkError


class TestMainErrors(unittest.TestCase):
    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace())
    @patch('mo.main.driver', side_effect=FrameworkError('FW ERROR MESSAGE'))
    def test_FrameworkError(self, mock_argparse, mock_driver):
        with self.assertLogs() as logger:
            main(argparse.ArgumentParser(), 'framework_string')
            self.assertEqual(logger.output, ['ERROR:root:FW ERROR MESSAGE'])

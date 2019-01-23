"""
 Copyright (c) 2018 Intel Corporation

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

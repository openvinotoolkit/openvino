"""
 Copyright (C) 2018-2020 Intel Corporation

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

import unittest
import unittest.mock as mock
from unittest.mock import mock_open
from unittest.mock import patch

from mo.utils.version import get_version


class TestingVersion(unittest.TestCase):
    def test_unknown_version(self):
        self.assertEqual(get_version(), "unknown version")

    @patch('os.path.isfile')
    @mock.patch('builtins.open', new_callable=mock_open, create=True, read_data='2021.1.0-1028-55e4d5673a8')
    def test_get_version(self, mock_open, mock_isfile):
        mock_isfile.return_value = True
        mock_open.return_value.__enter__ = mock_open
        self.assertEqual(get_version(), '2021.1.0-1028-55e4d5673a8')
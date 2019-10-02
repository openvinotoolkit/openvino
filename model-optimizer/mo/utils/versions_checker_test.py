"""
 Copyright (c) 2019 Intel Corporation

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
from mo.utils.versions_checker import get_module_version_list_from_file, parse_versions_list

class TestingVersionsChecker(unittest.TestCase):
    @mock.patch('builtins.open', new_callable=mock_open, create=True)
    def test_get_module_version_list_from_file(self, mock_open):
        mock_open.return_value.__enter__ = mock_open
        mock_open.return_value.__iter__ = mock.Mock(
            return_value=iter(['mxnet>=1.0.0,<=1.3.1', 'networkx>=1.11', 'numpy==1.12.0', 'defusedxml<=0.5.0']))
        ref_list =[('mxnet', '>=', '1.0.0'), ('mxnet', '<=', '1.3.1'),
                          ('networkx', '>=', '1.11'),
                          ('numpy', '==', '1.12.0'), ('defusedxml', '<=', '0.5.0')]
        version_list = get_module_version_list_from_file('mock_file')
        self.assertEqual(len(version_list), 5)
        for i, version_dict in enumerate(version_list):
            self.assertTupleEqual(ref_list[i], version_dict)

    @mock.patch('builtins.open', new_callable=mock_open, create=True)
    def test_get_module_version_list_from_file_with_fw_name(self, mock_open):
        mock_open.return_value.__enter__ = mock_open
        mock_open.return_value.__iter__ = mock.Mock(
            return_value=iter(['mxnet']))
        ref_list = [('mxnet', None, None)]
        version_list = get_module_version_list_from_file('mock_file')
        self.assertEqual(len(version_list), 1)
        for i, version_dict in enumerate(version_list):
            self.assertTupleEqual(ref_list[i], version_dict)

    def test_append_version_list(self):
        v1 = 'mxnet>=1.0.0,<=1.3.1'
        req_list = list()
        parse_versions_list(v1, req_list)
        ref_list = [('mxnet', '>=', '1.0.0'),
                    ('mxnet', '<=', '1.3.1')]
        for i, v in enumerate(req_list):
            self.assertEqual(v, ref_list[i])

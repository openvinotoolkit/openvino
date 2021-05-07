# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import unittest.mock as mock
from unittest.mock import mock_open

from mo.utils.versions_checker import get_module_version_list_from_file, parse_and_filter_versions_list


class TestingVersionsChecker(unittest.TestCase):
    @mock.patch('builtins.open', new_callable=mock_open, create=True)
    def test_get_module_version_list_from_file(self, mock_open):
        mock_open.return_value.__enter__ = mock_open
        mock_open.return_value.__iter__ = mock.Mock(
            return_value=iter(['mxnet>=1.0.0,<=1.3.1', 'networkx>=1.11', 'numpy==1.12.0', 'defusedxml<=0.5.0']))
        ref_list =[('mxnet', '>=', '1.0.0'), ('mxnet', '<=', '1.3.1'),
                          ('networkx', '>=', '1.11'),
                          ('numpy', '==', '1.12.0'), ('defusedxml', '<=', '0.5.0')]
        version_list = get_module_version_list_from_file('mock_file', {})
        self.assertEqual(len(version_list), 5)
        for i, version_dict in enumerate(version_list):
            self.assertTupleEqual(ref_list[i], version_dict)

    @mock.patch('builtins.open', new_callable=mock_open, create=True)
    def test_get_module_version_list_from_file2(self, mock_open):
        mock_open.return_value.__enter__ = mock_open
        mock_open.return_value.__iter__ = mock.Mock(
            return_value=iter(['tensorflow>=1.15.2,<2.0; python_version < "3.8"',
                               'tensorflow>=2.0; python_version >= "3.8"',
                               'numpy==1.12.0',
                               'defusedxml<=0.5.0']))
        ref_list =[('tensorflow', '>=', '1.15.2'),
                   ('tensorflow', '<', '2.0'),
                   ('numpy', '==', '1.12.0'),
                   ('defusedxml', '<=', '0.5.0')]
        version_list = get_module_version_list_from_file('mock_file', {'python_version': '3.7.0'})
        self.assertEqual(len(version_list), 4)
        for i, version_dict in enumerate(version_list):
            self.assertTupleEqual(ref_list[i], version_dict)

    @mock.patch('builtins.open', new_callable=mock_open, create=True)
    def test_get_module_version_list_from_file3(self, mock_open):
        mock_open.return_value.__enter__ = mock_open
        mock_open.return_value.__iter__ = mock.Mock(
            return_value=iter(['tensorflow>=1.15.2,<2.0; python_version < "3.8"',
                               'tensorflow>=2.0; python_version >= "3.8"',
                               'numpy==1.12.0',
                               'defusedxml<=0.5.0']))
        ref_list =[('tensorflow', '>=', '2.0'),
                   ('numpy', '==', '1.12.0'),
                   ('defusedxml', '<=', '0.5.0')]
        version_list = get_module_version_list_from_file('mock_file', {'python_version': '3.8.1'})
        self.assertEqual(len(version_list), 3)
        for i, version_dict in enumerate(version_list):
            self.assertTupleEqual(ref_list[i], version_dict)

    @mock.patch('builtins.open', new_callable=mock_open, create=True)
    def test_get_module_version_list_from_file_with_fw_name(self, mock_open):
        mock_open.return_value.__enter__ = mock_open
        mock_open.return_value.__iter__ = mock.Mock(
            return_value=iter(['mxnet']))
        ref_list = [('mxnet', None, None)]
        version_list = get_module_version_list_from_file('mock_file', {})
        self.assertEqual(len(version_list), 1)
        for i, version_dict in enumerate(version_list):
            self.assertTupleEqual(ref_list[i], version_dict)

    def test_append_version_list(self):
        v1 = 'mxnet>=1.0.0,<=1.3.1'
        req_list = list()
        parse_and_filter_versions_list(v1, req_list, {})
        ref_list = [('mxnet', '>=', '1.0.0'),
                    ('mxnet', '<=', '1.3.1')]
        for i, v in enumerate(req_list):
            self.assertEqual(v, ref_list[i])

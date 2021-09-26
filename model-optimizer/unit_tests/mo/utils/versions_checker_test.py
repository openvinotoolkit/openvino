# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import unittest.mock as mock
from unittest.mock import mock_open

from mo.utils.versions_checker import get_module_version_list_from_file, parse_and_filter_versions_list, version_check


class TestingVersionsChecker(unittest.TestCase):
    @mock.patch('builtins.open', new_callable=mock_open, create=True)
    def test_get_module_version_list_from_file(self, mock_open):
        mock_open.return_value.__enter__ = mock_open
        mock_open.return_value.__iter__ = mock.Mock(
            return_value=iter(['mxnet>=1.0.0,<=1.3.1',
                               'networkx>=1.11',
                               'numpy==1.12.0',
                               'defusedxml<=0.5.0',
                               'networkx~=1.11']))
        ref_list = [('mxnet', '>=', '1.0.0'),
                    ('mxnet', '<=', '1.3.1'),
                    ('networkx', '>=', '1.11'),
                    ('numpy', '==', '1.12.0'),
                    ('defusedxml', '<=', '0.5.0'),
                    ('networkx', '~=', '1.11')]
        version_list = get_module_version_list_from_file('mock_file', {})
        self.assertEqual(len(version_list), 6)
        for i, version_dict in enumerate(version_list):
            self.assertTupleEqual(ref_list[i], version_dict)

    @mock.patch('builtins.open', new_callable=mock_open, create=True)
    def test_get_module_version_list_from_file2(self, mock_open):
        mock_open.return_value.__enter__ = mock_open
        mock_open.return_value.__iter__ = mock.Mock(
            return_value=iter(['tensorflow>=1.15.2,<2.0; python_version < "3.8"',
                               'tensorflow>=2.0; python_version >= "3.8"',
                               'numpy==1.12.0',
                               'defusedxml<=0.5.0',
                               'networkx~=1.11']))
        ref_list = [('tensorflow', '>=', '1.15.2'),
                    ('tensorflow', '<', '2.0'),
                    ('numpy', '==', '1.12.0'),
                    ('defusedxml', '<=', '0.5.0'),
                    ('networkx', '~=', '1.11')]
        version_list = get_module_version_list_from_file('mock_file', {'python_version': '3.7.0'})
        self.assertEqual(len(version_list), 5)
        for i, version_dict in enumerate(version_list):
            self.assertTupleEqual(ref_list[i], version_dict)

    @mock.patch('builtins.open', new_callable=mock_open, create=True)
    def test_get_module_version_list_from_file3(self, mock_open):
        mock_open.return_value.__enter__ = mock_open
        mock_open.return_value.__iter__ = mock.Mock(
            return_value=iter(['# Commented line',
                               'tensorflow>=1.15.2,<2.0; python_version < "3.8"',
                               'tensorflow>=2.0; python_version >= "3.8" # Comment after line',
                               'numpy==1.12.0',
                               'defusedxml<=0.5.0',
                               'networkx~=1.11']))
        ref_list = [('tensorflow', '>=', '2.0'),
                    ('numpy', '==', '1.12.0'),
                    ('defusedxml', '<=', '0.5.0'),
                    ('networkx', '~=', '1.11')]
        version_list = get_module_version_list_from_file('mock_file', {'python_version': '3.8.1'})
        self.assertEqual(len(version_list), 4)
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
        req_list = []
        parse_and_filter_versions_list(v1, req_list, {})
        ref_list = [('mxnet', '>=', '1.0.0'),
                    ('mxnet', '<=', '1.3.1')]
        for i, v in enumerate(ref_list):
            self.assertEqual(v, req_list[i])

    def test_append_version_list_sys_neg_1(self):
        v1 = "mxnet>=1.7.0 ; sys_platform != 'win32'"
        req_list = []
        parse_and_filter_versions_list(v1, req_list, {'sys_platform': 'darwin'})
        ref_list = [('mxnet', '>=', '1.7.0')]
        for i, v in enumerate(ref_list):
            self.assertEqual(v, req_list[i])

    def test_append_version_list_sys_neg_2(self):
        v1 = "mxnet>=1.7.0 ; sys_platform != 'win32'"
        req_list = []
        parse_and_filter_versions_list(v1, req_list, {'sys_platform': 'win32'})
        ref_list = []
        for i, v in enumerate(ref_list):
            self.assertEqual(v, req_list[i])

    def test_append_version_list_sys(self):
        v1 = "mxnet>=1.7.0 ; sys_platform == 'linux'"
        req_list = []

        parse_and_filter_versions_list(v1, req_list, {'sys_platform': 'linux'})
        ref_list = [('mxnet', '>=', '1.7.0')]
        for i, v in enumerate(ref_list):
            self.assertEqual(v, req_list[i])

    def test_append_version_list_sys_double_quotes(self):
        v1 = "mxnet>=1.7.0 ; sys_platform == \"linux\""
        req_list = []

        parse_and_filter_versions_list(v1, req_list, {'sys_platform': 'linux'})
        ref_list = [('mxnet', '>=', '1.7.0')]
        for i, v in enumerate(ref_list):
            self.assertEqual(v, req_list[i])

    def test_append_version_list_py_ver_single_quotes(self):
        v1 = "mxnet>=1.7.0 ; python_version < '3.8'"
        req_list = []

        parse_and_filter_versions_list(v1, req_list, {'python_version': '3.7.1'})
        ref_list = [('mxnet', '>=', '1.7.0')]
        for i, v in enumerate(ref_list):
            self.assertEqual(v, req_list[i])

    def test_append_version_list_sys_python_ver_1(self):
        v1 = "mxnet>=1.7.0 ; sys_platform == 'linux' or python_version >= \"3.8\""
        req_list = []
        parse_and_filter_versions_list(v1, req_list, {'python_version': '3.8.1', 'sys_platform': 'linux'})
        ref_list = []
        for i, v in enumerate(ref_list):
            self.assertEqual(v, req_list[i])

    def test_append_version_list_sys_python_ver_2(self):
        v1 = "mxnet>=1.7.0 ; sys_platform == 'linux' and python_version >= \"3.8\""
        req_list = []
        parse_and_filter_versions_list(v1, req_list, {'python_version': '3.7.1', 'sys_platform': 'linux'})
        ref_list = []
        for i, v in enumerate(ref_list):
            self.assertEqual(v, req_list[i])

    def test_version_check_equal(self):
        modules_versions_list = [('module_1', '==', '2.0', '2.0'),
                                 ('module_2', '==', '2.0', '2.0.1'),
                                 ]

        ref_list = [('module_2', 'installed: 2.0.1', 'required: == 2.0'),
                    ]

        not_satisfied_versions = []

        for name, key, required_version, installed_version in modules_versions_list:
            version_check(name, installed_version, required_version, key, not_satisfied_versions)
        self.assertEqual(not_satisfied_versions, ref_list)

    def test_version_check_less_equal(self):
        modules_versions_list = [('module_1', '>=', '1.12.0', '1.09.2'),
                                 ('module_2', '>=', '1.12.0', '1.12.0'),
                                 ('module_3', '>=', '1.12.0', '1.12.1'),
                                 ('module_4', '>=', '1.12.0', '1.20.0'),
                                 ]

        ref_list = [('module_1', 'installed: 1.09.2', 'required: >= 1.12.0'),
                    ]

        not_satisfied_versions = []

        for name, key, required_version, installed_version in modules_versions_list:
            version_check(name, installed_version, required_version, key, not_satisfied_versions)
        self.assertEqual(not_satisfied_versions, ref_list)

    def test_version_check_greater_equal(self):
        modules_versions_list = [('module_1', '>=', '1.12.0', '1.09.2'),
                                 ('module_2', '>=', '1.12.0', '1.12.0'),
                                 ('module_3', '>=', '1.12.0', '1.12.1'),
                                 ('module_4', '>=', '1.12.0', '1.20.0'),
                                 ('module_5', '>=', '1.12.0.post2', '1.12.0.post1'),
                                 ]

        ref_list = [('module_1', 'installed: 1.09.2', 'required: >= 1.12.0'),
                    ('module_5', 'installed: 1.12.0.post1', 'required: >= 1.12.0.post2')
                    ]

        not_satisfied_versions = []

        for name, key, required_version, installed_version in modules_versions_list:
            version_check(name, installed_version, required_version, key, not_satisfied_versions)
        self.assertEqual(not_satisfied_versions, ref_list)

    def test_version_check_less(self):
        modules_versions_list = [('module_1', '<', '1.11', '1.01'),
                                 ('module_2', '<', '1.11', '1.10.1'),
                                 ('module_3', '<', '1.11', '1.11'),
                                 ('module_4', '<', '1.11', '1.20'),
                                 ('module_5', '<', '1.11', '1.10.post2'),
                                 ]

        ref_list = [('module_3', 'installed: 1.11', 'required: < 1.11'),
                    ('module_4', 'installed: 1.20', 'required: < 1.11'),
                    ]

        not_satisfied_versions = []

        for name, key, required_version, installed_version in modules_versions_list:
            version_check(name, installed_version, required_version, key, not_satisfied_versions)
        self.assertEqual(not_satisfied_versions, ref_list)

    def test_version_check_greater(self):
        modules_versions_list = [('module_1', '>', '1.11', '1.01'),
                                 ('module_2', '>', '1.11', '1.11'),
                                 ('module_3', '>', '1.11', '1.11.1'),
                                 ('module_4', '>', '1.11', '1.20'),
                                 ]

        ref_list = [('module_1', 'installed: 1.01', 'required: > 1.11'),
                    ('module_2', 'installed: 1.11', 'required: > 1.11'),
                    ]

        not_satisfied_versions = []

        for name, key, required_version, installed_version in modules_versions_list:
            version_check(name, installed_version, required_version, key, not_satisfied_versions)
        self.assertEqual(not_satisfied_versions, ref_list)

    def test_version_check_compatible(self):
        modules_versions_list = [('module_1', '~=', '1.2.3', '1.2.3'),
                                 ('module_2', '~=', '1.2.3', '1.2.3b4'),
                                 ('module_3', '~=', '1.2.3', '1.2.4'),
                                 ('module_4', '~=', '1.2.3', '1.3.0'),
                                 ('module_5', '~=', '1.2.3', '1.2.2'),
                                 ('module_6', '~=', '1.2.3', '2.2.2'),
                                 ('module_7', '~=', '1.2.post2', '1.3'),
                                 ('module_8', '~=', '1.2.post2', '1.2'),
                                 ('module_9', '~=', '1.2.post2', '1.2.post1'),
                                 ('module_9', '~=', '1.2.post2', '1.2.post4')
                                 ]

        ref_list = [('module_4', 'installed: 1.3.0', 'required: ~= 1.2.3'),
                    ('module_5', 'installed: 1.2.2', 'required: ~= 1.2.3'),
                    ('module_6', 'installed: 2.2.2', 'required: ~= 1.2.3'),
                    ('module_8', 'installed: 1.2', 'required: ~= 1.2.post2'),
                    ('module_9', 'installed: 1.2.post1', 'required: ~= 1.2.post2')
                    ]

        not_satisfied_versions = []

        for name, key, required_version, installed_version in modules_versions_list:
            version_check(name, installed_version, required_version, key, not_satisfied_versions)
        self.assertEqual(not_satisfied_versions, ref_list)

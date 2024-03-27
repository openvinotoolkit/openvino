# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import unittest
import unittest.mock as mock
from unittest.mock import mock_open
from unittest.mock import patch

from openvino.tools.mo.subprocess_main import setup_env
from openvino.tools.mo.utils.version import get_version, extract_release_version, get_simplified_ie_version, \
    get_simplified_mo_version, extract_hash_from_version, VersionChecker


class TestingVersion(unittest.TestCase):
    @patch('os.path.isfile')
    @mock.patch('builtins.open', new_callable=mock_open, create=True, read_data='2021.1.0-1028-55e4d5673a8')
    def test_get_version(self, mock_open, mock_isfile):
        mock_isfile.return_value = True
        mock_open.return_value.__enter__ = mock_open
        self.assertEqual(get_version(), '2021.1.0-1028-55e4d5673a8')

    @patch('os.path.isfile')
    @mock.patch('builtins.open', new_callable=mock_open, create=True, read_data='2021.1.0-1028-55e4d5673a8')
    def test_release_version_extractor(self, mock_open, mock_isfile):
        mock_isfile.return_value = True
        mock_open.return_value.__enter__ = mock_open
        self.assertEqual(extract_release_version(get_version()), ('2021', '1'))

    @patch('os.path.isfile')
    @mock.patch('builtins.open', new_callable=mock_open, create=True, read_data='custom_releases/2021/1_55e4d5673a8')
    def test_custom_release_version_extractor(self, mock_open, mock_isfile):
        mock_isfile.return_value = True
        mock_open.return_value.__enter__ = mock_open
        self.assertEqual(extract_release_version(get_version()), ('2021', '1'))

    @patch('os.path.isfile')
    @mock.patch('builtins.open', new_callable=mock_open, create=True, read_data='custom_my_branch/fix_55e4d5673a8')
    def test_release_version_extractor_neg(self, mock_open, mock_isfile):
        mock_isfile.return_value = True
        mock_open.return_value.__enter__ = mock_open
        self.assertEqual(extract_release_version(get_version()), (None, None))

    @patch('os.path.isfile')
    @mock.patch('builtins.open', new_callable=mock_open, create=True, read_data='custom_releases/2021/1_55e4d5673a8')
    def test_simplify_mo_version_release(self, mock_open, mock_isfile):
        mock_isfile.return_value = True
        mock_open.return_value.__enter__ = mock_open
        self.assertEqual(get_simplified_mo_version(), "2021.1")

    @patch('os.path.isfile')
    @mock.patch('builtins.open', new_callable=mock_open, create=True, read_data='custom_my_branch/fix_55e4d5673a8')
    def test_simplify_mo_version_custom(self, mock_open, mock_isfile):
        mock_isfile.return_value = True
        mock_open.return_value.__enter__ = mock_open
        self.assertEqual(get_simplified_mo_version(), "custom")

    def test_simplify_ie_version_release_legacy(self):
        self.assertEqual(get_simplified_ie_version(version="2.1.custom_releases/2021/3_4c8eae"), "2021.3")

    def test_simplify_ie_version_release(self):
        self.assertEqual(get_simplified_ie_version(version="custom_releases/2021/3_4c8eae"), "2021.3")

    def test_simplify_ie_version_custom_legacy(self):
        self.assertEqual(get_simplified_ie_version(version="2.1.custom_my/branch/3_4c8eae"), "custom")

    def test_simplify_ie_version_custom(self):
        self.assertEqual(get_simplified_ie_version(version="custom_my/branch/3_4c8eae"), "custom")

    def test_extracting_version_hash_full_with_build_number(self):
        self.assertEqual(extract_hash_from_version(full_version="2021.1.0-1028-55e4d5673a8"), "55e4d5673a8")

    def test_extracting_version_hash_full_with_build_number_dirty(self):
        self.assertEqual(extract_hash_from_version(full_version="2021.1.0-1028-55e4d5673a8-dirty"), "55e4d5673a8")

    def test_extracting_version_hash_full_with_build_number_private(self):
        self.assertEqual(extract_hash_from_version(full_version="2021.1.0-1028-55e4d5673a8-private"), "55e4d5673a8")

    def test_extracting_version_hash_custom_master(self):
        self.assertEqual(extract_hash_from_version(full_version="custom_master_55e4d5673a833abab638ee9837bc87a0b7c3a043"),
                         "55e4d5673a833abab638ee9837bc87a0b7c3a043")

    def test_extracting_version_hash_mo_format(self):
        self.assertEqual(extract_hash_from_version(full_version="2022.1.custom_master_55e4d5673a833abab638ee9837bc87a0b7c3a043"),
                         "55e4d5673a833abab638ee9837bc87a0b7c3a043")

    def test_negative_extracting_version_hash(self):
        self.assertEqual(extract_hash_from_version(full_version="2022.1.custom_master"),
                         None)

    # format from the current nightly wheel
    def test_extracting_version_hash_from_old_format(self):
        self.assertEqual(extract_hash_from_version(full_version="2022.1.0-6311-a90bb1f"),
                         "a90bb1f")

    def test_version_checker(self):
        setup_env()
        args = [sys.executable, '-m', 'pytest',
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'convert/version_checker_test_actual.py'), '-s']

        status = subprocess.run(args, env=os.environ, capture_output=True)
        assert not status.returncode

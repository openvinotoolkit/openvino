# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import re
import unittest
from itertools import islice

from openvino.tools.ovc.utils import get_mo_root_dir

dir_patterns_to_skip = ['.*__pycache__.*']
file_patterns_to_skip = ['.*\\.DS_Store$',
                         '.*\\.swp',
                         '.*\\.pyc$',
                         'requirements.*\.txt',
                         'version.txt']
full_name_patterns_to_skip = []
if platform.system() == 'Windows':
    full_name_patterns_to_skip = [i.replace('/', '\\\\') for i in full_name_patterns_to_skip]


def is_match(name: str, patterns: ()):
    return any((re.match(pattern, name) for pattern in patterns))


class TestBOMFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.existing_files = []
        cur_path = os.path.join(os.path.realpath(__file__), os.pardir)
        mo_path = os.path.abspath(os.path.join(cur_path, os.pardir, os.pardir))
        with open(os.path.join(mo_path, 'unit_tests', 'ovc', 'package_BOM.txt'), 'r') as bom_file:
            if platform.system() == 'Windows':
                cls.existing_files = [name.rstrip().replace('/', '\\') for name in bom_file.readlines()]
            else:
                cls.existing_files = [name.rstrip() for name in bom_file.readlines()]

        # dirs_to_search is the root directory where MO is located, 'openvino_project_root/tools/mo/openvino/tools'
        cls.dirs_to_search = os.path.normpath(get_mo_root_dir() + '/ovc/')
        cls.prefix = os.path.normpath(get_mo_root_dir() + '../../../')  # prefix which is used in BOM file
        cls.expected_header = [re.compile(pattern) for pattern in [
            r'^# Copyright \([cC]\) [0-9\-]+ Intel Corporation$',
            r'^# SPDX-License-Identifier: Apache-2.0$',
        ]]

    def test_bom_file(self):
        missing_files = list()
        for src_dir in [self.dirs_to_search]:
            if not os.path.isdir(src_dir):
                continue
            for root, dirs, files in os.walk(src_dir):
                if is_match(root, dir_patterns_to_skip):
                    continue
                for f in files:
                    full_name = os.path.join(root, f)
                    full_name = full_name[len(self.prefix) + 1:]
                    if is_match(f, file_patterns_to_skip):
                        continue
                    if is_match(full_name, full_name_patterns_to_skip):
                        continue
                    if full_name not in self.existing_files:
                        missing_files.append(full_name)

        if len(missing_files) != 0:
            print("Missing files:")
            for f in missing_files:
                print(f.replace('\\', '/'))
        self.assertTrue(not len(missing_files), '{} files missed in BOM'.format(len(missing_files)))

    def test_bom_does_not_contain_unittest_files(self):
        for file_name in self.existing_files:
            self.assertFalse(file_name.endswith('_test.py'), 'BOM file contains test file {}'.format(file_name))

    def test_deleted_files_still_stored_in_bom(self):
        deleted = list()
        for file in self.existing_files:
            if not os.path.isfile(os.path.join(self.prefix, file)):
                deleted.append(file)
        if len(deleted) != 0:
            print("Deleted files still stored in BOM file:")
            for f in deleted:
                print(f)
        self.assertTrue(not len(deleted), '{} files deleted but still stored in BOM'.format(len(deleted)))

    def test_alphabetical_order_and_duplicates(self):
        sorted_bom = sorted([x for x in self.existing_files if self.existing_files.count(x) == 1], key=str.lower)
        if self.existing_files != sorted_bom:
            print("Wrong order. Alphabetical order of BOM is:")
            print(*sorted_bom, sep='\n')
            self.assertTrue(False)

    def test_missed_intel_header(self):
        missing_files = list()
        for src_dir in [self.dirs_to_search]:
            if not os.path.isdir(src_dir):
                continue
            for root, dirs, files in os.walk(src_dir):
                if is_match(root, dir_patterns_to_skip):
                    continue
                for f in files:
                    ignores = [
                        '^__init__.py$',
                        '^caffe_pb2.py$',
                        '^.*.pyc$',
                        '^generate_caffe_pb2.py$'
                    ]
                    if not is_match(f, ['.*.py$']) or is_match(f, ignores):
                        continue
                    full_name = os.path.join(root, f)
                    with open(full_name, 'r') as source_f:
                        # read two more lines from the file because it can contain shebang and empty lines
                        s = [x.strip() for x in islice(source_f, len(self.expected_header) + 2)]
                        # skip shebang and empty lines in the beginning of the file
                        try:
                            while s[0] in ('', '#!/usr/bin/env python3'):
                                s = s[1:]
                            for str_ind in range(0, len(self.expected_header)):
                                if not re.match(self.expected_header[str_ind], s[str_ind]):
                                    missing_files.append(full_name)
                                    break
                        except:
                            pass
        self.assertTrue(not len(missing_files),
                        '{} files with missed header: \n{}'.format(len(missing_files), '\n'.join(missing_files)))

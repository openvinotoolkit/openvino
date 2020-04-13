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

import os
import platform
import re
import unittest
from itertools import islice

dir_patterns_to_skip = ['.*__pycache__.*']
file_patterns_to_skip = ['.*_test\\.py$',
                         '.*\\.DS_Store$',
                         '.*\\.swp',
                         '.*\\.pyc$']
full_name_patterns_to_skip = ['^mo/utils/unittest/.*\.py$',
                              '^mo/utils/convert.py$',
                              '^extensions/front/caffe/CustomLayersMapping.xml$',
                              '^mo/utils/unittest/test_data/.*\.xml$',
                              '^mo/utils/unittest/test_data/.*\.bin$',
                              ]
if platform.system() == 'Windows':
    full_name_patterns_to_skip = [i.replace('/', '\\\\') for i in full_name_patterns_to_skip]
dirs_to_search = ['mo', 'extensions', 'tf_call_ie_layer']


def is_match(name: str, patterns: ()):
    return any((re.match(pattern, name) for pattern in patterns))


class TestBOMFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.existing_files = []
        cur_path = os.path.join(os.path.realpath(__file__), os.pardir)
        cls.output_dir = os.path.abspath(os.path.join(cur_path, os.pardir))
        with open(os.path.join(cls.output_dir, 'automation', 'package_BOM.txt'), 'r') as bom_file:
            if platform.system() == 'Windows':
                cls.existing_files = [name.rstrip().replace('/', '\\') for name in bom_file.readlines()]
            else:
                cls.existing_files = [name.rstrip() for name in bom_file.readlines()]

        cls.expected_header = [re.compile(pattern) for pattern in [
            '^"""$',
            '^Copyright \([cC]\) [0-9\-]+ Intel Corporation$',
            '^$',
            '^Licensed under the Apache License, Version 2.0 \(the "License"\);$',
            '^you may not use this file except in compliance with the License.$',
            '^You may obtain a copy of the License at$',
            '^$',
            '^http://www.apache.org/licenses/LICENSE-2.0$',
            '^$',
            '^Unless required by applicable law or agreed to in writing, software$',
            '^distributed under the License is distributed on an "AS IS" BASIS,$',
            '^WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.$',
            '^See the License for the specific language governing permissions and$',
            '^limitations under the License.$',
            '^"""$'
        ]]

    def test_bom_file(self):
        missing_files = list()
        for src_dir in dirs_to_search:
            src_dir = os.path.join(self.output_dir, src_dir)
            if not os.path.isdir(src_dir):
                continue
            for root, dirs, files in os.walk(src_dir):
                if is_match(root, dir_patterns_to_skip):
                    continue
                for f in files:
                    full_name = os.path.join(root, f)
                    full_name = full_name[len(self.output_dir) + 1:]
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
            if not os.path.isfile(os.path.join(self.output_dir, file)):
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
        for src_dir in dirs_to_search:
            src_dir = os.path.join(self.output_dir, src_dir)
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
                        while s[0] in ('', '#!/usr/bin/env python3'):
                            s = s[1:]
                        for str_ind in range(0, len(self.expected_header)):
                            if not re.match(self.expected_header[str_ind], s[str_ind]):
                                missing_files.append(full_name)
                                break
        self.assertTrue(not len(missing_files),
                        '{} files with missed header: \n{}'.format(len(missing_files), '\n'.join(missing_files)))

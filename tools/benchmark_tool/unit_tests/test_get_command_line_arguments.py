# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.benchmark.utils.utils import get_command_line_arguments


class TestGetCommandLineArguments(unittest.TestCase):
    def test_separate_name_and_value(self):
        self.assertEqual(get_command_line_arguments(['benchmark_app', '-m', 'model.xml', '-niter', '10']),
                         [('-m', 'model.xml'), ('-niter', '10')])

    def test_name_and_value_joined_by_equals(self):
        self.assertEqual(get_command_line_arguments(['benchmark_app', '-m=model.xml']),
                         [('-m', 'model.xml')])

    def test_value_containing_equals_is_kept_whole(self):
        # only the first '=' separates the name from the value
        self.assertEqual(get_command_line_arguments(['benchmark_app', '-m=/data/exp=1/model.xml']),
                         [('-m', '/data/exp=1/model.xml')])

    def test_separate_value_containing_equals_is_a_value(self):
        # a value that is not an option must not be split, even though it contains '='
        self.assertEqual(get_command_line_arguments(['benchmark_app', '-m', '/data/exp=1/model.xml', '-niter', '10']),
                         [('-m', '/data/exp=1/model.xml'), ('-niter', '10')])

    def test_separate_config_value_containing_equals(self):
        self.assertEqual(get_command_line_arguments(['benchmark_app', '-load_config', 'CPU:{NUM_STREAMS=4}']),
                         [('-load_config', 'CPU:{NUM_STREAMS=4}')])

    def test_empty_argument_is_not_fatal(self):
        # an unset shell variable expands to an empty argument
        self.assertEqual(get_command_line_arguments(['benchmark_app', '-m', 'model.xml', '-i', '']),
                         [('-m', 'model.xml'), ('-i', '')])


if __name__ == '__main__':
    unittest.main()

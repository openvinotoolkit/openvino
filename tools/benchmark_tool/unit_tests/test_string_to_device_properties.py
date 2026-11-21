# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.benchmark.utils.utils import string_to_device_properties


class TestStringToDeviceProperties(unittest.TestCase):
    def test_empty_string_gives_no_properties(self):
        self.assertEqual(string_to_device_properties(''), {})

    def test_properties_per_device(self):
        self.assertEqual(string_to_device_properties('{CPU:{NUM_STREAMS:2},GPU:{NUM_STREAMS:4}}'),
                         {'CPU': {'NUM_STREAMS': '2'}, 'GPU': {'NUM_STREAMS': '4'}})

    def test_value_without_enclosing_braces_is_reported(self):
        # the message itself contains braces, it must not be broken by str.format()
        cases = [
            ('CPU:{NUM_STREAMS:2}', 'C', '}'),  # no opening brace
            ('{CPU:{NUM_STREAMS:2', '{', '2'),  # no closing brace
            ('CPU:NUM_STREAMS:2', 'C', '2'),    # neither
        ]
        for value, first, last in cases:
            with self.subTest(value=value):
                with self.assertRaises(Exception) as context:
                    string_to_device_properties(value)
                self.assertNotIsInstance(context.exception, KeyError)
                self.assertEqual(
                    str(context.exception),
                    "Failed to parse device properties. Value of device properties should be started with '{' "
                    f"and ended with '}}'. They are actually {first} and {last}")


if __name__ == '__main__':
    unittest.main()

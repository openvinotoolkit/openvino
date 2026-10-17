# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.benchmark.utils.utils import parse_value_per_device


class TestParseValuePerDevice(unittest.TestCase):
    DEVICES = ['CPU', 'GPU']

    def test_per_device_value(self):
        self.assertEqual(parse_value_per_device(self.DEVICES, 'CPU:4', 'nstreams'),
                         {'CPU': '4'})

    def test_value_applied_to_all_devices(self):
        self.assertEqual(parse_value_per_device(self.DEVICES, '4', 'nstreams'),
                         {'CPU': '4', 'GPU': '4'})

    def test_several_devices(self):
        self.assertEqual(parse_value_per_device(self.DEVICES, 'CPU:4,GPU:8', 'nstreams'),
                         {'CPU': '4', 'GPU': '8'})

    def test_empty_string(self):
        self.assertEqual(parse_value_per_device(self.DEVICES, '', 'nstreams'), {})

    def test_unknown_device_is_rejected(self):
        with self.assertRaises(Exception):
            parse_value_per_device(self.DEVICES, 'NPU:4', 'nstreams')

    def test_extra_colon_is_rejected(self):
        # must not be silently dropped
        with self.assertRaises(Exception):
            parse_value_per_device(self.DEVICES, 'CPU:4:8', 'nstreams')

    def test_extra_colon_is_rejected_among_valid_values(self):
        with self.assertRaises(Exception):
            parse_value_per_device(self.DEVICES, 'CPU:4,GPU:8:16', 'nstreams')


if __name__ == '__main__':
    unittest.main()

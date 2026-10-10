# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.benchmark.utils.utils import parse_devices


class TestParseDevices(unittest.TestCase):
    def test_single_device(self):
        self.assertEqual(parse_devices('CPU'), ['CPU'])

    def test_virtual_device_with_hw_devices(self):
        self.assertEqual(parse_devices('MULTI:CPU,GPU'), ['MULTI', 'CPU', 'GPU'])

    def test_excluded_device(self):
        self.assertEqual(parse_devices('AUTO:-CPU'), ['AUTO', 'CPU'])

    def test_trailing_comma(self):
        self.assertEqual(parse_devices('HETERO:CPU,'), ['HETERO', 'CPU'])

    def test_repeated_comma(self):
        self.assertEqual(parse_devices('MULTI:CPU,,GPU'), ['MULTI', 'CPU', 'GPU'])


if __name__ == '__main__':
    unittest.main()

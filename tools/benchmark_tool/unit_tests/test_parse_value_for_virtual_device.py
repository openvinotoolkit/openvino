# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.benchmark.utils.utils import parse_devices, parse_value_per_device, \
    parse_value_for_virtual_device


def parse(device_string, values_string):
    """Mirror the -d/-nstreams handling in benchmark main.py."""
    devices = parse_devices(device_string)
    parsed = parse_value_per_device(devices, values_string, 'nstreams')
    if 'AUTO' in devices or 'MULTI' in devices or 'HETERO' in devices:
        parse_value_for_virtual_device(device_string.partition(':')[0], parsed)
    return parsed


class TestParseValueForVirtualDevice(unittest.TestCase):
    def test_auto_keeps_its_own_value(self):
        # -nstreams 2 with -d AUTO:CPU is applied to AUTO and must not be dropped.
        self.assertEqual(parse('AUTO:CPU', '2'), {'AUTO': '2'})

    def test_hetero_keeps_its_own_value(self):
        self.assertEqual(parse('HETERO:CPU', '2'), {'HETERO': '2'})

    def test_auto_with_several_hardware_devices(self):
        self.assertEqual(parse('AUTO:CPU,GPU', '2'), {'AUTO': '2'})

    def test_auto_alone_keeps_its_own_value(self):
        # The virtual device's own entry must survive instead of being deleted.
        self.assertEqual(parse('AUTO', '2'), {'AUTO': '2'})

    def test_multi_wraps_values_for_hardware_devices(self):
        self.assertEqual(parse('MULTI:CPU', '2'), {'MULTI': '{CPU:2}'})

    def test_per_device_value_is_wrapped_for_auto(self):
        self.assertEqual(parse('AUTO:CPU', 'CPU:2'), {'AUTO': '{CPU:2}'})

    def test_single_device_is_untouched(self):
        self.assertEqual(parse('CPU', '2'), {'CPU': '2'})


if __name__ == '__main__':
    unittest.main()

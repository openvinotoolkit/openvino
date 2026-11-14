# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openvino import OVAny, properties
from openvino.tools.benchmark.utils.utils import (
    device_properties_to_string,
    dump_config,
    load_config,
    string_to_device_properties,
)


class TestDeviceProperties(unittest.TestCase):
    def test_empty_properties(self):
        self.assertEqual(device_properties_to_string({}), '{}')
        self.assertEqual(string_to_device_properties('{}'), {})

    def test_empty_nested_properties(self):
        self.assertEqual(device_properties_to_string({'CPU': {}}), '{CPU:{}}')
        self.assertEqual(string_to_device_properties('{CPU:{}}'), {'CPU': {}})

    def test_empty_and_nonempty_properties(self):
        config = {'CPU': {}, 'GPU': {'PERF_COUNT': True, 'NUM_STREAMS': 2}}
        serialized = '{CPU:{},GPU:{PERF_COUNT:YES,NUM_STREAMS:2}}'
        self.assertEqual(device_properties_to_string(config), serialized)
        self.assertEqual(string_to_device_properties(serialized),
                         {'CPU': {}, 'GPU': {'PERF_COUNT': 'YES', 'NUM_STREAMS': '2'}})

    def test_dump_load_empty_device_properties(self):
        for device_properties in ({}, {'CPU': {}}):
            with (
                self.subTest(device_properties=device_properties),
                TemporaryDirectory() as directory,
            ):
                filename = Path(directory) / 'config.json'
                key = properties.device.properties()
                dump_config(filename, {'AUTO': {key: OVAny(device_properties)}})
                loaded = {}
                load_config(filename, loaded)
                self.assertEqual(loaded['AUTO'][key].value, device_properties)


if __name__ == '__main__':
    unittest.main()

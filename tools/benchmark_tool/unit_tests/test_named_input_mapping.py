# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np
from openvino import Layout, PartialShape
from openvino.tools.benchmark.utils.inputs_filling import parse_path
from openvino.tools.benchmark.utils.utils import AppInputInfo


class TestNamedInputMapping(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'input.npy'
        np.save(self.path, np.zeros((1, 3, 2, 2), dtype=np.float32))

    def input_info(self, name, node_name=None):
        info = AppInputInfo()
        info.name = name
        info.node_name = node_name if node_name is not None else name
        info.layout = Layout('NCHW')
        info.partial_shape = PartialShape([1, 3, 2, 2])
        return info

    def test_tensor_names_are_matched_literally(self):
        for name in ['input', 'x', 'input.0', 'input-0', 'model/input', 'input:0', 'input[0]', 'input$']:
            with self.subTest(name=name):
                images, arrays, binaries = parse_path(f'{name}:{self.path}', [self.input_info(name)])
                self.assertEqual(dict(arrays), {name: [str(self.path)]})
                self.assertFalse(images)
                self.assertFalse(binaries)

    def test_node_name_maps_to_the_tensor_name(self):
        info = self.input_info('tensor:0', 'model/parameter-0')
        _, arrays, _ = parse_path(f'{info.node_name}:{self.path}', [info])
        self.assertEqual(dict(arrays), {info.name: [str(self.path)]})

    def test_longest_tensor_name_is_matched_first(self):
        other_path = self.path.with_name('other.npy')
        np.save(other_path, np.ones((1, 3, 2, 2), dtype=np.float32))
        inputs = [self.input_info('input'), self.input_info('input:0')]
        _, arrays, _ = parse_path(f'input:0:{self.path},input:{other_path}', inputs)
        self.assertEqual(dict(arrays), {'input:0': [str(self.path)], 'input': [str(other_path)]})

    def test_unmapped_path_is_preserved(self):
        _, arrays, _ = parse_path(str(self.path), [self.input_info('input')])
        self.assertEqual(dict(arrays), {'input': [str(self.path)]})

    def test_unknown_mapping_name_is_rejected(self):
        with self.assertRaisesRegex(Exception, 'Wrong input mapping! Cannot find inputs:'):
            parse_path(f'missing:{self.path}', [self.input_info('input')])

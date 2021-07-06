# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import numpy as np

from mo.front.common.partial_infer.elemental import copy_shape_infer


class FakeNode:
    def __init__(self, blob):
        self.blob = blob

    def in_shape(self):
        return self.blob


class TestElementalInference(unittest.TestCase):
    @patch('mo.front.common.partial_infer.elemental.single_output_infer')
    def test_copy_shape_infer(self, single_output_infer_mock):
        single_output_infer_mock.return_value = 0
        node = FakeNode(np.array([1, 2]))
        copy_shape_infer(node)
        self.assertTrue(single_output_infer_mock.called)

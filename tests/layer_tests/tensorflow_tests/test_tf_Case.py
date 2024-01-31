# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestCaseFloat(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'cond' in inputs_info, "Test error: inputs_info must contain `cond`"
        assert 'x' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y' in inputs_info, "Test error: inputs_info must contain `y`"
        cond_shape = inputs_info['cond']
        x_shape = inputs_info['x']
        y_shape = inputs_info['y']
        inputs_data = {}
        inputs_data['cond'] = np.random.randint(0, 2, cond_shape).astype(bool)
        inputs_data['x'] = np.random.randint(1, 10, x_shape).astype(np.float32)
        inputs_data['y'] = np.random.randint(-50, 50, y_shape).astype(np.float32)
        return inputs_data

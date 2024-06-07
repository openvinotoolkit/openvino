# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from jax_layer_test_class import JaxLayerTest
import jax.numpy as jnp

class TestAdd(JaxLayerTest):
    def _prepare_input(self):
        lhs = np.random.randint(-10, 10, self.lhs_shape).astype(self.input_type)
        rhs = np.random.randint(-10, 10, self.rhs_shape).astype(self.input_type)
        return (lhs, rhs)

    def create_model(self, lhs_shape, rhs_shape, input_type):
        self.lhs_shape = lhs_shape
        self.rhs_shape = rhs_shape
        self.input_type = input_type

        def jax_add(lhs, rhs):
            return lhs + rhs

        return jax_add, None

    test_data = [
        dict(lhs_shape=[4], rhs_shape=[4]),
        dict(lhs_shape=[2, 5], rhs_shape=[5]),
        dict(lhs_shape=[2, 3, 4, 5], rhs_shape=[4, 2, 5, 3]),
    ]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("input_type", [np.float32, np.int32])
    def test_dot_general(self, ie_device, precision, ir_version, params, input_type):
        self._test(*self.create_model(**params, input_type=input_type), ie_device, precision,
                   ir_version)

class TestAddWithConstant(JaxLayerTest):
    def _prepare_input(self):
        lhs = np.random.randint(-10, 10, self.lhs_shape).astype(self.input_type)
        rhs = np.random.randint(-10, 10, self.rhs_shape).astype(self.input_type)
        return (lhs, rhs)

    def create_model(self, lhs_shape, rhs_shape, input_type):
        self.lhs_shape = lhs_shape
        self.rhs_shape = rhs_shape
        self.input_type = input_type

        def jax_add_with_constant(lhs, rhs):
            const = jnp.ones((), dtype=self.input_type)
            return lhs + rhs + const

        return jax_add_with_constant, None

    test_data = [
        dict(lhs_shape=[4], rhs_shape=[4]),
        dict(lhs_shape=[2, 5], rhs_shape=[5]),
        dict(lhs_shape=[2, 3, 4, 5], rhs_shape=[4, 2, 5, 3]),
    ]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("input_type", [np.float32, np.int32])
    def test_dot_general(self, ie_device, precision, ir_version, params, input_type):
        self._test(*self.create_model(**params, input_type=input_type), ie_device, precision,
                   ir_version)

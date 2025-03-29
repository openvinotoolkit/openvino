# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest

from jax_layer_test_class import JaxLayerTest


def get_rand_jnp_array(shape, dtype):
    if dtype in [np.float32, np.float16, np.float64]:
        return jnp.array(np.random.uniform(-1000, 1000, shape).astype(dtype))
    else:
        return jnp.array(np.random.randint(-10, 10, shape).astype(dtype))


class TestMul(JaxLayerTest):
    def _prepare_input(self):
        lhs = get_rand_jnp_array(self.lhs_shape, self.lhs_type)
        rhs = get_rand_jnp_array(self.rhs_shape, self.rhs_type)
        return (lhs, rhs)

    def create_model(self, lhs_shape, rhs_shape, lhs_type, rhs_type):
        self.lhs_shape = lhs_shape
        self.rhs_shape = rhs_shape
        self.lhs_type = lhs_type
        self.rhs_type = rhs_type

        def jax_mul(lhs, rhs):
            return lhs * rhs

        return jax_mul, None, 'mul'

    test_data = [
        dict(lhs_shape=[4], rhs_shape=[4]),
        dict(lhs_shape=[2, 5], rhs_shape=[2, 5]),
        dict(lhs_shape=[2, 3, 4, 5], rhs_shape=[2, 3, 4, 5]),
    ]

    input_types = [
        (np.float32, np.float32),
        (np.int32, np.float32),
        (np.float32, np.float16),
        (np.int32, np.int64),
    ]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("input_type", input_types)
    def test_mul(self, ie_device, precision, ir_version, params, input_type):
        self._test(*self.create_model(**params, lhs_type=input_type[0], rhs_type=input_type[1]), ie_device, precision,
                   ir_version)


class TestMulWithConstant(JaxLayerTest):
    def _prepare_input(self):
        lhs = get_rand_jnp_array(self.lhs_shape, self.lhs_type)
        rhs = get_rand_jnp_array(self.rhs_shape, self.rhs_type)
        return (lhs, rhs)

    def create_model(self, lhs_shape, rhs_shape, lhs_type, rhs_type):
        self.lhs_shape = lhs_shape
        self.rhs_shape = rhs_shape
        self.lhs_type = lhs_type
        self.rhs_type = rhs_type
        self.const = jnp.ones(self.lhs_shape, dtype=self.lhs_type)

        def jax_mul_with_constant(lhs, rhs):
            return lhs * rhs * self.const

        return jax_mul_with_constant, None, 'mul'

    test_data = [
        dict(lhs_shape=[4], rhs_shape=[4]),
        dict(lhs_shape=[2, 5], rhs_shape=[2, 5]),
        dict(lhs_shape=[2, 3, 4, 5], rhs_shape=[2, 3, 4, 5]),
    ]

    input_types = [
        (np.float32, np.float32),
        (np.int32, np.float32),
        (np.float32, np.float16),
        (np.int32, np.int64),
    ]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("input_type", input_types)
    def test_mul_with_constant(self, ie_device, precision, ir_version, params, input_type):
        self._test(*self.create_model(**params, lhs_type=input_type[0], rhs_type=input_type[1]), ie_device, precision,
                   ir_version)


class TestMulWithLiteralInvar(JaxLayerTest):
    def _prepare_input(self):
        lhs = get_rand_jnp_array(self.lhs_shape, self.lhs_type)
        rhs = get_rand_jnp_array(self.rhs_shape, self.rhs_type)
        return (lhs, rhs)

    def create_model(self, lhs_shape, rhs_shape, lhs_type, rhs_type):
        self.lhs_shape = lhs_shape
        self.rhs_shape = rhs_shape
        self.lhs_type = lhs_type
        self.rhs_type = rhs_type

        def jax_mul_with_constant(lhs, rhs):
            x = lhs * 5
            return x * rhs

        return jax_mul_with_constant, None, 'mul'

    test_data = [
        dict(lhs_shape=[4], rhs_shape=[4]),
        dict(lhs_shape=[2, 5], rhs_shape=[2, 5]),
        dict(lhs_shape=[2, 3, 4, 5], rhs_shape=[2, 3, 4, 5]),
    ]

    input_types = [
        (np.float32, np.float32),
        (np.int32, np.float32),
        (np.float32, np.float16),
        (np.int32, np.int64),
    ]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("input_type", input_types)
    def test_mul_with_literal_invar(self, ie_device, precision, ir_version, params, input_type):
        self._test(*self.create_model(**params, lhs_type=input_type[0], rhs_type=input_type[1]), ie_device, precision,
                   ir_version)

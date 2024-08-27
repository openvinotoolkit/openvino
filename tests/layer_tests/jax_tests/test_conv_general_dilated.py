# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest


class TestConvGeneralDilated(JaxLayerTest):
    def _prepare_input(self):
        lhs = np.random.rand(*self.lhs_shape).astype(np.float32)
        return [lhs]

    def create_model(self, lhs_shape, rhs_shape, window_strides, padding,
                     lhs_dilation, dimension_numbers,
                     feature_group_count):
        self.lhs_shape = lhs_shape
        kernel = jnp.array(np.random.rand(*rhs_shape), dtype=jnp.float32)

        def jax_conv_general_dilated(lhs):
            out = lax.conv_general_dilated(lhs=lhs, rhs=kernel, window_strides=window_strides, padding=padding,
                                           lhs_dilation=lhs_dilation, dimension_numbers=dimension_numbers,
                                           feature_group_count=feature_group_count)
            return out

        return jax_conv_general_dilated, None, 'conv_general_dilated'

    test_data_basic = [
        # regular convolution with NCHW layout for inputs and NHWC layout for output
        dict(lhs_shape=[2, 3, 40, 60], rhs_shape=[4, 3, 2, 3],
             dimension_numbers=('NCHW', 'OIHW', 'NHWC'), feature_group_count=1),
        # group convolution with groups = 3
        dict(lhs_shape=[2, 3 * 4, 20, 30], rhs_shape=[3 * 2, 4, 2, 2],
             dimension_numbers=('NCHW', 'OIHW', 'NHWC'), feature_group_count=3),
        # regular convolution with NHWC layout for input and NCHW layout for output
        dict(lhs_shape=[1, 30, 20, 3], rhs_shape=[4, 3, 2, 3],
             dimension_numbers=('NHWC', 'OIHW', 'NCHW'), feature_group_count=1),
    ]

    @pytest.mark.parametrize("padding", [
        'SAME_LOWER', 'SAME', 'VALID'
    ])
    @pytest.mark.parametrize("window_strides", [
        [1, 1], [1, 2], [3, 2]
    ])
    @pytest.mark.parametrize("lhs_dilation", [
        None, [1, 1],
        # other type of lhs dilation is not supported by TF for tracing
        # https://github.com/google/jax/issues/4216
    ])
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_conv_general_dilated(self, ie_device, precision, ir_version, params, padding, window_strides,
                                  lhs_dilation):
        self._test(*self.create_model(**params, padding=padding,
                                      window_strides=window_strides, lhs_dilation=lhs_dilation),
                   ie_device, precision,
                   ir_version)

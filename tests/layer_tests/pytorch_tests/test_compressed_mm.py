# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


def pack_uint4(tensor):
    packed_tensor = tensor.contiguous()
    packed_tensor = packed_tensor.reshape(-1, 2)
    packed_tensor = torch.bitwise_and(packed_tensor[..., ::2], 15) | packed_tensor[..., 1::2] << 4
    return packed_tensor


def unpack_uint4(packed_tensor):
    return torch.stack((torch.bitwise_and(packed_tensor, 15), torch.bitwise_right_shift(packed_tensor, 4)), dim=-1)


def pack_int4(tensor):
    tensor = tensor + 8
    return pack_uint4(tensor.type(torch.uint8))


def unpack_int4(packed_tensor):
    t = unpack_uint4(packed_tensor)
    return t.type(torch.int8) - 8


def decompress_asymmetric(input, scale, zero_point):
    input = input.type(dtype=scale.dtype)
    zero_point = zero_point.type(dtype=scale.dtype)
    decompressed_input = (input - zero_point) * scale
    return decompressed_input


def decompress_symmetric(input, scale):
    input = input.type(dtype=scale.dtype)
    decompressed_input = input * scale
    return decompressed_input


class TestMatMulU4Weights(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (np.round(5.00 * self.rng.random([2, 4], dtype=np.float32) - 2.50, 4),)

    def create_model(self, group_size):
        class aten_mm_u4(torch.nn.Module):
            def __init__(self, compressed_weight, scale, zero_point, weight_shape):
                super(aten_mm_u4, self).__init__()
                self.compressed_weight_shape = compressed_weight.shape
                self.packed_weight = torch.nn.Parameter(pack_uint4(compressed_weight), requires_grad=False)

                self.register_buffer("_scale", scale.type(dtype=torch.float16))

                self.zero_point_shape = zero_point.shape
                self.register_buffer("_zero_point", pack_uint4(zero_point))

                self.weight_shape = weight_shape

            def forward(self, x):
                # NNCF UINT4 asymmetric decompression pattern
                # https://github.com/openvinotoolkit/nncf/blob/develop/nncf/torch/quantization/layers.py
                compressed_weight = unpack_uint4(self.packed_weight)
                compressed_weight = compressed_weight.reshape(self.compressed_weight_shape)

                zero_point = unpack_uint4(self._zero_point)
                zero_point = zero_point.reshape(self.zero_point_shape)

                weight = decompress_asymmetric(compressed_weight, self._scale, zero_point)
                weight = weight.reshape(self.weight_shape)
                weight = weight.type(dtype=torch.float32)

                return torch.matmul(x, weight)

        ref_net = None

        weight_shape = (4, 2)
        ngroups = weight_shape[0] // group_size
        compressed_weight_shape = (ngroups, group_size, weight_shape[1])
        zero_point_shape = scale_shape = (ngroups, 1, weight_shape[1])

        compressed_weight = (15.00 * self.rng.random(compressed_weight_shape, dtype=np.float32)).astype(dtype=np.uint8)
        scale = np.round(10.00 * self.rng.random(scale_shape, dtype=np.float32) - 5.00)
        zero_point = (15.00 * self.rng.random(zero_point_shape, dtype=np.float32)).astype(dtype=np.uint8)

        t_compressed_weight = torch.from_numpy(compressed_weight)
        t_scale = torch.from_numpy(scale)
        t_zero_point = torch.from_numpy(zero_point)

        return aten_mm_u4(t_compressed_weight, t_scale, t_zero_point, weight_shape), ref_net, ["aten::matmul"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("group_size", [2, 4])
    def test_matmul_u4(self, group_size, ie_device, precision, ir_version):
        self._test(
            *self.create_model(group_size),
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            dynamic_quantization_group_size=0
        )


class TestMatMulI4Weights(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (np.round(5.00 * self.rng.random([2, 4], dtype=np.float32) - 2.50, 4),)

    def create_model(self, group_size):
        class aten_mm_i4(torch.nn.Module):
            def __init__(self, compressed_weight, scale, weight_shape):
                super(aten_mm_i4, self).__init__()
                self.compressed_weight_shape = compressed_weight.shape
                self.packed_weight = torch.nn.Parameter(pack_int4(compressed_weight), requires_grad=False)

                self.register_buffer("_scale", scale.type(dtype=torch.float16))

                self.weight_shape = weight_shape

            def forward(self, x):
                # NNCF INT4 symmetric decompression pattern
                # https://github.com/openvinotoolkit/nncf/blob/develop/nncf/torch/quantization/layers.py
                compressed_weight = unpack_int4(self.packed_weight)
                compressed_weight = compressed_weight.reshape(self.compressed_weight_shape)

                weight = decompress_symmetric(compressed_weight, self._scale)
                weight = weight.reshape(self.weight_shape)
                weight = weight.type(dtype=torch.float32)

                return torch.matmul(x, weight)

        ref_net = None

        weight_shape = (4, 2)
        ngroups = weight_shape[0] // group_size
        compressed_weight_shape = (ngroups, group_size, weight_shape[1])
        scale_shape = (ngroups, 1, weight_shape[1])

        compressed_weight = (16.00 * self.rng.random(compressed_weight_shape, dtype=np.float32) - 8.00).astype(
            dtype=np.int8
        )
        scale = np.round(10.00 * self.rng.random(scale_shape, dtype=np.float32) - 5.00)

        t_compressed_weight = torch.from_numpy(compressed_weight)
        t_scale = torch.from_numpy(scale)

        return aten_mm_i4(t_compressed_weight, t_scale, weight_shape), ref_net, ["aten::matmul"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("group_size", [2, 4])
    def test_matmul_i4(self, group_size, ie_device, precision, ir_version):
        self._test(
            *self.create_model(group_size),
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            dynamic_quantization_group_size=0
        )

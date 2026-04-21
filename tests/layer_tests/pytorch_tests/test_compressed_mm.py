# Copyright (C) 2018-2026 Intel Corporation
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


def pack_uint3(tensor):
    packed_tensor = tensor.contiguous().reshape(-1, 8)
    
    b0 = (packed_tensor[:, 0] << 5) | (packed_tensor[:, 1] << 2) | (packed_tensor[:, 2] >> 1)
    b1 = (packed_tensor[:, 2] << 7) | (packed_tensor[:, 3] << 4) | (packed_tensor[:, 4] << 1) | (packed_tensor[:, 5] >> 2)
    b2 = (packed_tensor[:, 5] << 6) | (packed_tensor[:, 6] << 3) | (packed_tensor[:, 7])

    # Stack and flatten to get a 1D packed byte array
    return torch.stack([b0, b1, b2], dim=1).reshape(-1)


def unpack_uint3(packed_tensor):
    bytes_3 = packed_tensor.view(-1, 3)
    b0, b1, b2 = bytes_3[:, 0], bytes_3[:, 1], bytes_3[:, 2]

    # Extract 3-bit values
    # w0 = (b0 >> 5) & 7
    # w1 = (b0 >> 2) & 7
    # w2 = ((b0 << 1) & 6) | ((b1 >> 7) & 1)  # Combine 2 bits from b0, 1 bit from b1
    # w3 = (b1 >> 4) & 7
    # w4 = (b1 >> 1) & 7
    # w5 = ((b1 << 2) & 4) | ((b2 >> 6) & 3)  # Combine 1 bit from b1, 2 bits from b2
    # w6 = (b2 >> 3) & 7
    # w7 = b2 & 7

    # w0 = torch.bitwise_and(torch.bitwise_right_shift(b0, 5), 7)
    # w1 = torch.bitwise_and(torch.bitwise_right_shift(b0, 2), 7)
    # w2 = torch.bitwise_or(torch.bitwise_and(torch.bitwise_left_shift(b0, 1), 6),
    #                       torch.bitwise_and(torch.bitwise_right_shift(b1, 7), 1))
    # w3 = torch.bitwise_and(torch.bitwise_right_shift(b1, 4), 7)
    # w4 = torch.bitwise_and(torch.bitwise_right_shift(b1, 1), 7)
    # w5 = torch.bitwise_or(torch.bitwise_and(torch.bitwise_left_shift(b1, 2), 6),
    #                       torch.bitwise_and(torch.bitwise_right_shift(b2, 6), 3))
    # w6 = torch.bitwise_and(torch.bitwise_right_shift(b2, 3), 7)
    # w7 = torch.bitwise_and(b2, 7)

    return torch.stack((
            torch.bitwise_and(b0 >> 5, 7),
            torch.bitwise_and(b0 >> 2, 7),
            torch.bitwise_or(torch.bitwise_and(b0 << 1, 6),
                                  torch.bitwise_and(b1 >> 7, 1)),
            torch.bitwise_and(b1 >> 4, 7),
            torch.bitwise_and(b1 >> 1, 7),
            torch.bitwise_or(torch.bitwise_and(b1 << 2, 6),
                                  torch.bitwise_and(b2 >> 6, 3)),
            torch.bitwise_and(b2 >> 3, 7),
            torch.bitwise_and(b2, 7)),
            dim=-1)

    # return torch.stack((
    #         torch.bitwise_and(torch.bitwise_right_shift(b0, 5), 7),
    #         torch.bitwise_and(torch.bitwise_right_shift(b0, 2), 7),
    #         torch.bitwise_or(torch.bitwise_and(torch.bitwise_left_shift(b0, 1), 6),
    #                               torch.bitwise_and(torch.bitwise_right_shift(b1, 7), 1)),
    #         torch.bitwise_and(torch.bitwise_right_shift(b1, 4), 7),
    #         torch.bitwise_and(torch.bitwise_right_shift(b1, 1), 7),
    #         torch.bitwise_or(torch.bitwise_and(torch.bitwise_left_shift(b1, 2), 6),
    #                               torch.bitwise_and(torch.bitwise_right_shift(b2, 6), 3)),
    #         torch.bitwise_and(torch.bitwise_right_shift(b2, 3), 7),
    #         torch.bitwise_and(b2, 7)),
    #         dim=-1)

    w0 = torch.bitwise_and(b0, 7)
    w1 = torch.bitwise_and(b0, 3)
    w2 = torch.bitwise_or(b1, 1)
    w3 = torch.bitwise_and(b1, 7)
    w4 = torch.bitwise_and(b1, 4)
    w5 = torch.bitwise_or(b2, 3)
    w6 = torch.bitwise_and(b2, 7)
    w7 = torch.bitwise_and(b2, 7)

    # Stack and restore original shape
    unpacked = torch.stack((w0, w1, w2, w3, w4, w5, w6, w7), dim=1)
    return unpacked.reshape(-1)


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

    def _prepare_input(self):
        return (np.round(5.00 * self.random.rand(2, 4) - 2.50, 4),)

    def create_model(self, group_size):
        class aten_mm_u4(torch.nn.Module):
            def __init__(self, compressed_weight, scale, zero_point, weight_shape):
                super().__init__()
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


        weight_shape = (4, 2)
        ngroups = weight_shape[0] // group_size
        compressed_weight_shape = (ngroups, group_size, weight_shape[1])
        zero_point_shape = scale_shape = (ngroups, 1, weight_shape[1])

        compressed_weight = self.random.randint(0, 16, size=compressed_weight_shape, dtype=np.uint8)
        scale = self.random.randint(-5, 6, size=scale_shape, dtype=np.int32)
        zero_point = self.random.randint(0, 16, size=zero_point_shape, dtype=np.uint8)

        t_compressed_weight = torch.from_numpy(compressed_weight)
        t_scale = torch.from_numpy(scale)
        t_zero_point = torch.from_numpy(zero_point)

        return aten_mm_u4(t_compressed_weight, t_scale, t_zero_point, weight_shape), ["aten::matmul"]

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

    def _prepare_input(self):
        return (np.round(5.00 * self.random.rand(2, 4) - 2.50, 4),)

    def create_model(self, group_size):
        class aten_mm_i4(torch.nn.Module):
            def __init__(self, compressed_weight, scale, weight_shape):
                super().__init__()
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


        weight_shape = (4, 2)
        ngroups = weight_shape[0] // group_size
        compressed_weight_shape = (ngroups, group_size, weight_shape[1])
        scale_shape = (ngroups, 1, weight_shape[1])

        compressed_weight = self.random.randint(-8, 8, size=compressed_weight_shape, dtype=np.int8)
        scale = self.random.randint(-5, 6, size=scale_shape, dtype=np.int32)

        t_compressed_weight = torch.from_numpy(compressed_weight)
        t_scale = torch.from_numpy(scale)

        return aten_mm_i4(t_compressed_weight, t_scale, weight_shape), ["aten::matmul"]

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


class TestMatMulU3Weights(PytorchLayerTest):
    def _prepare_input(self):
        return (np.round(5.00 * self.random.rand(2, 32) - 2.50, 4),)

    def create_model(self, group_size):
        class aten_mm_u3(torch.nn.Module):
            ZERO_POINT_VALUE = 2

            def __init__(self, compressed_weight, scale, weight_shape, result_dtype):
                super(aten_mm_u3, self).__init__()
                self.compressed_weight_shape = compressed_weight.shape
                self.packed_weight = torch.nn.Parameter(
                    pack_uint3(compressed_weight), requires_grad=False
                )

                self.register_buffer("_scale", scale.type(dtype=torch.float16))
                self.register_buffer(
                    "_zero_point",
                    torch.tensor(self.ZERO_POINT_VALUE, dtype=torch.uint8),
                )

                self.weight_shape = weight_shape
                self.result_dtype = result_dtype

            def forward(self, x):
                # NNCF INT3 symmetric decompression pattern
                compressed_weight = unpack_uint3(self.packed_weight)
                compressed_weight = compressed_weight.reshape(
                    self.compressed_weight_shape
                )

                compressed_weight = compressed_weight.type(
                    dtype=self.result_dtype
                ) - self._zero_point.type(dtype=self.result_dtype)

                weight = decompress_symmetric(compressed_weight, self._scale)
                weight = weight.reshape(self.weight_shape)
                weight = weight.type(dtype=torch.float32)

                return torch.matmul(x, weight)

        weight_shape = (32, 4)
        ngroups = weight_shape[0] // group_size
        compressed_weight_shape = (ngroups, group_size, weight_shape[1])
        scale_shape = (ngroups, 1, weight_shape[1])

        compressed_weight = self.random.randint(
            0, 8, size=compressed_weight_shape, dtype=np.uint8
        )
        scale = self.random.randint(-5, 6, size=scale_shape, dtype=np.int32)

        t_compressed_weight = torch.from_numpy(compressed_weight)
        t_scale = torch.from_numpy(scale)

        return aten_mm_u3(t_compressed_weight, t_scale, weight_shape, torch.float32), [
            "aten::matmul"
        ]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("group_size", [8, 16])
    def test_matmul_u3(self, group_size, ie_device, precision, ir_version):
        self._test(
            *self.create_model(group_size),
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            dynamic_quantization_group_size=0,
        )

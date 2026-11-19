// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vm/mem_ref_type.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "openvino/runtime/make_tensor.hpp"

namespace {

class NpuMemRefTypeTest : public testing::TestWithParam<ov::element::Type> {};

TEST_P(NpuMemRefTypeTest, SetsContiguousElementStrides) {
    const auto tensor = ov::make_tensor(GetParam(), ov::Shape{2, 3});
    if (GetParam().bitwidth() < 8) {
        EXPECT_THROW(tensor->get_strides(), ov::Exception);
    }

    intel_npu::MemRefType memref;
    ASSERT_NO_THROW(memref.set(tensor->data(), 0, tensor));
    EXPECT_EQ(memref._sizes, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(memref._strides, (std::vector<int64_t>{3, 1}));
    EXPECT_EQ(memref._data, tensor->data());
}

INSTANTIATE_TEST_SUITE_P(HostCompile,
                         NpuMemRefTypeTest,
                         testing::Values(ov::element::u4, ov::element::i4, ov::element::nf4, ov::element::f16));

TEST(NpuMemRefStridedTensorTest, PreservesNonContiguousF16Strides) {
    std::vector<uint16_t> storage(16);
    const auto tensor = ov::make_tensor(ov::element::f16, ov::Shape{2, 3}, storage.data(), ov::Strides{16, 4});
    intel_npu::MemRefType memref;
    ASSERT_NO_THROW(memref.set(tensor->data(), 0, tensor));
    EXPECT_EQ(memref._strides, (std::vector<int64_t>{8, 2}));
    EXPECT_EQ(memref._data, storage.data());
}

class NpuMemRefStridedSubByteTest : public testing::TestWithParam<ov::element::Type> {};

TEST_P(NpuMemRefStridedSubByteTest, PreservesNonContiguousSubByteStrides) {
    const auto type = GetParam();
    constexpr size_t row_elements = 16;
    const size_t dense_row_bytes = row_elements * type.bitwidth() / 8;
    const size_t padded_row_bytes = 2 * dense_row_bytes;
    std::vector<uint8_t> storage(2 * padded_row_bytes);

    const auto tensor =
        ov::make_tensor(type, ov::Shape{2, row_elements}, storage.data(), ov::Strides{padded_row_bytes, 1});

    intel_npu::MemRefType memref;
    ASSERT_NO_THROW(memref.set(tensor->data(), 0, tensor));
    EXPECT_EQ(memref._sizes, (std::vector<int64_t>{2, static_cast<int64_t>(row_elements)}));
    EXPECT_EQ(memref._strides, (std::vector<int64_t>{static_cast<int64_t>(padded_row_bytes * 8 / type.bitwidth()), 1}));
    EXPECT_EQ(memref._data, storage.data());
}

INSTANTIATE_TEST_SUITE_P(HostCompile,
                         NpuMemRefStridedSubByteTest,
                         testing::Values(ov::element::u1,
                                         ov::element::u2,
                                         ov::element::u4,
                                         ov::element::i4,
                                         ov::element::nf4,
                                         ov::element::f4e2m1));

}  // namespace

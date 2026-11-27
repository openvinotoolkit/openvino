// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov::intel_cpu;

TEST(GenerateCpuBlockedMemoryDescTest, DenseTensorUsesRowMajorStrides) {
    std::vector<float> buffer(6);
    ov::Tensor tensor(ov::element::f32, ov::Shape{2, 3}, buffer.data());

    const auto desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(ov::get_tensor_impl(tensor));

    EXPECT_EQ(desc->getStrides(), VectorDims({3, 1}));
}

TEST(GenerateCpuBlockedMemoryDescTest, StridedTensorConvertsByteStridesToElementStrides) {
    // Each row holds 3 real floats but is padded to 4 floats (16 bytes).
    std::vector<float> buffer(2 * 4);
    ov::Tensor tensor(ov::element::f32, ov::Shape{2, 3}, buffer.data(), ov::Strides{16, 4});

    const auto desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(ov::get_tensor_impl(tensor));

    EXPECT_EQ(desc->getStrides(), VectorDims({4, 1}));
}

TEST(GenerateCpuBlockedMemoryDescTest, StridedTensorAvoidsOverflowConvertingLargeByteStrides) {
    constexpr auto shift = std::numeric_limits<size_t>::digits - 3;
    std::vector<float> buffer(2);
    const ov::Strides huge_byte_strides{size_t{1} << shift, sizeof(float)};
    ov::Tensor tensor(ov::element::f32, ov::Shape{2, 1}, buffer.data(), huge_byte_strides);

    const auto desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(ov::get_tensor_impl(tensor));

    EXPECT_EQ(desc->getStrides(), VectorDims({size_t{1} << (shift - 2), 1}));
}

TEST(GenerateCpuBlockedMemoryDescTest, NonUnitInnermostStrideIsPreservedForByteGranularType) {
    // Every other f16 element is used: innermost element stride is 2, not 1.
    std::vector<uint16_t> buffer(2 * 8);
    ov::Tensor tensor(ov::element::f16, ov::Shape{2, 3}, buffer.data(), ov::Strides{16, 4});

    const auto desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(ov::get_tensor_impl(tensor));

    EXPECT_EQ(desc->getStrides(), VectorDims({8, 2}));
}

TEST(GenerateCpuBlockedMemoryDescTest, DenseSubByteTensorUsesRowMajorStrides) {
    // u4: 100 elements per row, byte-aligned (100 * 4 bits == 50 bytes).
    std::vector<uint8_t> buffer(2 * 50);
    ov::Tensor tensor(ov::element::u4, ov::Shape{2, 100}, buffer.data());

    const auto desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(ov::get_tensor_impl(tensor));

    EXPECT_EQ(desc->getStrides(), VectorDims({100, 1}));
}

TEST(GenerateCpuBlockedMemoryDescTest, StridedSubByteTensorConvertsByteStridesToElementStrides) {
    // u4: 100 real elements (50 bytes) per row, padded to 64 bytes per row.
    std::vector<uint8_t> buffer(2 * 64);
    ov::Tensor tensor(ov::element::u4, ov::Shape{2, 100}, buffer.data(), ov::Strides{64, 1});

    const auto desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(ov::get_tensor_impl(tensor));

    // 64 bytes == 128 u4 elements per row.
    EXPECT_EQ(desc->getStrides(), VectorDims({128, 1}));
}

TEST(GenerateCpuBlockedMemoryDescTest, UnalignedSubByteShapeFallsBackToRowMajorStrides) {
    // u4 with an odd last-dimension count (3 * 4 bits == 12 bits) is not byte-aligned, so byte
    // strides are not representable and dense element strides are used instead.
    std::vector<uint8_t> buffer(3);
    ov::Tensor tensor(ov::element::u4, ov::Shape{2, 3}, buffer.data());

    const auto desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(ov::get_tensor_impl(tensor));

    EXPECT_EQ(desc->getStrides(), VectorDims({3, 1}));
}

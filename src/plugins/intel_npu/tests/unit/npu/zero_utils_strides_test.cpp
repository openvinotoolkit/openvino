// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/except.hpp"

namespace ov::test {

using intel_npu::zeroUtils::byte_strides_to_element_strides, intel_npu::zeroUtils::get_capacity_size;

TEST(ZeroUtilsStridesTest, ByteGranularTypeKeepsNonUnitInnermostStride) {
    EXPECT_EQ(byte_strides_to_element_strides(ov::Strides{16, 4}, ov::element::f16), (ov::Strides{8, 2}));
    EXPECT_EQ(byte_strides_to_element_strides(ov::Strides{24, 8, 4}, ov::element::f32), (ov::Strides{6, 2, 1}));
    EXPECT_EQ(byte_strides_to_element_strides(ov::Strides{3, 1}, ov::element::u8), (ov::Strides{3, 1}));
}

TEST(ZeroUtilsStridesTest, SubByteTypeConvertsOuterStridesAndForcesInnermostToOne) {
    EXPECT_EQ(byte_strides_to_element_strides(ov::Strides{8, 1}, ov::element::u4), (ov::Strides{16, 1}));
    EXPECT_EQ(byte_strides_to_element_strides(ov::Strides{16, 8, 1}, ov::element::nf4), (ov::Strides{32, 16, 1}));
    EXPECT_EQ(byte_strides_to_element_strides(ov::Strides{4, 1}, ov::element::u2), (ov::Strides{16, 1}));
    EXPECT_EQ(byte_strides_to_element_strides(ov::Strides{2, 1}, ov::element::u1), (ov::Strides{16, 1}));
    EXPECT_EQ(byte_strides_to_element_strides(ov::Strides{8, 1}, ov::element::f4e2m1), (ov::Strides{16, 1}));
}

TEST(ZeroUtilsStridesTest, EmptyStridesStayEmpty) {
    EXPECT_TRUE(byte_strides_to_element_strides(ov::Strides{}, ov::element::u4).empty());
    EXPECT_TRUE(byte_strides_to_element_strides(ov::Strides{}, ov::element::f32).empty());
}

TEST(ZeroUtilsStridesTest, ThrowsWhenStrideIsNotMultipleOfElementSize) {
    EXPECT_THROW(byte_strides_to_element_strides(ov::Strides{6, 4}, ov::element::f32), ov::Exception);
}

TEST(ZeroUtilsStridesTest, CapacityOfByteGranularTensor) {
    EXPECT_EQ(get_capacity_size(ov::element::f32, ov::Shape{2, 3}, ov::Strides{16, 4}), 28u);
    EXPECT_EQ(get_capacity_size(ov::element::f32, ov::Shape{2, 3}, ov::Strides{12, 4}), 24u);
}

TEST(ZeroUtilsStridesTest, CapacityOfSubByteTensorUsesPackedLastDimension) {
    // Dense u4 {2, 16}: 2 rows * 8 bytes.
    EXPECT_EQ(get_capacity_size(ov::element::u4, ov::Shape{2, 16}, ov::Strides{8, 1}), 16u);
    // Padded rows (16 bytes apart): one full padded row plus one packed row.
    EXPECT_EQ(get_capacity_size(ov::element::u4, ov::Shape{2, 16}, ov::Strides{16, 1}), 24u);
    EXPECT_EQ(get_capacity_size(ov::element::u1, ov::Shape{3, 16}, ov::Strides{4, 1}), 10u);
    EXPECT_EQ(get_capacity_size(ov::element::u2, ov::Shape{2, 16}, ov::Strides{8, 1}), 12u);
}

}  // namespace

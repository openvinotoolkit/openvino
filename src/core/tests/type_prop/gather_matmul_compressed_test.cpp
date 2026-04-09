// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/gather_matmul_compressed.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using ov::op::v0::Constant;
using ov::op::v0::Parameter;

namespace ov::tests {

namespace {
auto make_const(element::Type et, const Shape& shape) {
    return Constant::create(et, shape, std::vector<float>(shape_size(shape), 0.f));
}

auto make_param(element::Type et, const PartialShape& shape) {
    return std::make_shared<Parameter>(et, shape);
}

auto make_empty_bias() {
    return std::make_shared<Constant>(element::dynamic, Shape{0});
}
}  // namespace

using GatherMatmulCompressedTest = TypePropOpTest<ov::op::internal::GatherMatmulCompressed>;

// ============================================================================
// Positive tests — GatherMatmulCompressed
// ============================================================================

// Basic 3D compressed weights
TEST_F(GatherMatmulCompressedTest, shape_basic_3d) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::u8, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_empty_bias();
    auto scales = make_const(element::f32, {8, 4096, 1});
    auto zp = make_const(element::u8, {8, 4096, 1});

    auto op = make_op(a, b, idx, bias, scales, zp);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

// 4D group-compressed weights with scales/zp
TEST_F(GatherMatmulCompressedTest, shape_grouped_4d) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::u8, {8, 4096, 16, 128});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_empty_bias();
    auto scales = make_const(element::f32, {8, 4096, 16, 1});
    auto zp = make_const(element::u8, {8, 4096, 16, 1});

    auto op = make_op(a, b, idx, bias, scales, zp);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
}

// ============================================================================
// Negative tests — GatherMatmulCompressed validation failures
// ============================================================================

TEST_F(GatherMatmulCompressedTest, fail_scales_not_constant) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::u8, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_empty_bias();
    auto scales = make_param(element::f32, {8, 4096, 1});  // Parameter, not Constant
    auto zp = make_const(element::u8, {8, 4096, 1});

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx, bias, scales, zp),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input weight_scales must be a Constant"));
}

TEST_F(GatherMatmulCompressedTest, fail_zp_not_constant) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::u8, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_empty_bias();
    auto scales = make_const(element::f32, {8, 4096, 1});
    auto zp = make_param(element::u8, {8, 4096, 1});  // Parameter, not Constant

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx, bias, scales, zp),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input weight_zero_points must be a Constant"));
}

// ============================================================================
// Clone tests
// ============================================================================

TEST_F(GatherMatmulCompressedTest, clone_preserves_output_shape) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::u8, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_empty_bias();
    auto scales = make_const(element::f32, {8, 4096, 1});
    auto zp = make_const(element::u8, {8, 4096, 1});

    auto op = make_op(a, b, idx, bias, scales, zp);
    auto cloned = op->clone_with_new_inputs({a, b, idx, bias, scales, zp});

    EXPECT_EQ(cloned->get_output_partial_shape(0), op->get_output_partial_shape(0));
    EXPECT_EQ(cloned->get_output_element_type(0), op->get_output_element_type(0));
}

}  // namespace ov::tests

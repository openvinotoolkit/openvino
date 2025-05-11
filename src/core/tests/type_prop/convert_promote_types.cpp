// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert_promote_types.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/parameter.hpp"

struct ConvertPromoteTypesTestParams {
    ov::PartialShape in0_shape;
    ov::element::Type in0_type;
    ov::PartialShape in1_shape;
    ov::element::Type in1_type;
    bool pytorch_scalar_promotion;
    bool promote_unsafe;
    ov::element::Type expected_type;
    ov::element::Type u64_integer;
};
class ConvertPromoteTypesTest : public TypePropOpTest<ov::op::v14::ConvertPromoteTypes>,
                                public testing::WithParamInterface<ConvertPromoteTypesTestParams> {};

TEST_F(ConvertPromoteTypesTest, default_ctor) {
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{});
    auto c = this->make_op();
    c->set_arguments(ov::OutputVector{in0, in1});
    c->set_pytorch_scalar_promotion(true);
    c->set_promote_unsafe(true);
    c->set_u64_integer_promotion_target(ov::element::f64);
    c->validate_and_infer_types();
    ASSERT_EQ(c->get_output_element_type(0), c->get_output_element_type(1));
    ASSERT_EQ(c->get_output_element_type(0), ov::element::f16);
    ASSERT_EQ(c->get_output_partial_shape(0), (ov::Shape{}));
    ASSERT_EQ(c->get_output_partial_shape(1), (ov::Shape{}));
}

TEST_P(ConvertPromoteTypesTest, suite) {
    auto& params = this->GetParam();
    auto in0_shape = params.in0_shape;
    if (!in0_shape.is_dynamic()) {
        set_shape_symbols(in0_shape);
    }
    auto in1_shape = params.in1_shape;
    if (!in1_shape.is_dynamic()) {
        set_shape_symbols(in1_shape);
    }

    auto in0 = std::make_shared<ov::op::v0::Parameter>(params.in0_type, in0_shape);
    auto in1 = std::make_shared<ov::op::v0::Parameter>(params.in1_type, in1_shape);
    auto c = this->make_op(in0, in1, params.pytorch_scalar_promotion, params.promote_unsafe, params.u64_integer);
    ASSERT_EQ(c->get_output_element_type(0), c->get_output_element_type(1));
    ASSERT_EQ(c->get_output_element_type(0), params.expected_type);
    ASSERT_EQ(c->get_output_partial_shape(0), (in0_shape));
    ASSERT_EQ(c->get_output_partial_shape(1), (in1_shape));
    ASSERT_EQ(get_shape_symbols(c->get_output_partial_shape(0)), get_shape_symbols(in0_shape));
    ASSERT_EQ(get_shape_symbols(c->get_output_partial_shape(1)), get_shape_symbols(in1_shape));
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         ConvertPromoteTypesTest,
                         testing::Values(
                             // Test cases:
                             //  dynamic
                             ConvertPromoteTypesTestParams{{1, 2, 3, 4},
                                                           ov::element::dynamic,
                                                           {5, 6, 7},
                                                           ov::element::dynamic,
                                                           false,
                                                           true,
                                                           ov::element::dynamic,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{5, 6, 7},
                                                           ov::element::dynamic,
                                                           {1},
                                                           ov::element::f16,
                                                           true,
                                                           false,
                                                           ov::element::dynamic,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1, 2, 3, 4},
                                                           ov::element::i64,
                                                           {1},
                                                           ov::element::dynamic,
                                                           true,
                                                           false,
                                                           ov::element::dynamic,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{},
                                                           ov::element::dynamic,
                                                           {1},
                                                           ov::element::f32,
                                                           true,
                                                           true,
                                                           ov::element::dynamic,
                                                           ov::element::f32},
                             //  bool
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::boolean,
                                                           {1, 2, 3, 4},
                                                           ov::element::boolean,
                                                           false,
                                                           true,
                                                           ov::element::boolean,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{ov::PartialShape().dynamic(),
                                                           ov::element::boolean,
                                                           {-1, {1, 5}, -1, -1},
                                                           ov::element::u32,
                                                           false,
                                                           true,
                                                           ov::element::u32,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::i16,
                                                           {1},
                                                           ov::element::boolean,
                                                           true,
                                                           false,
                                                           ov::element::i16,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::f16,
                                                           {1},
                                                           ov::element::boolean,
                                                           true,
                                                           false,
                                                           ov::element::f16,
                                                           ov::element::f32},
                             //  u and u
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u8,
                                                           {1},
                                                           ov::element::u8,
                                                           true,
                                                           false,
                                                           ov::element::u8,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u8,
                                                           {1},
                                                           ov::element::u4,
                                                           true,
                                                           false,
                                                           ov::element::u8,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u1,
                                                           {1},
                                                           ov::element::u4,
                                                           true,
                                                           false,
                                                           ov::element::u4,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u1,
                                                           {1},
                                                           ov::element::u1,
                                                           true,
                                                           false,
                                                           ov::element::u1,
                                                           ov::element::f32},
                             //  i and i
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::i8,
                                                           {1},
                                                           ov::element::i8,
                                                           true,
                                                           false,
                                                           ov::element::i8,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::i4,
                                                           {1},
                                                           ov::element::i8,
                                                           true,
                                                           false,
                                                           ov::element::i8,
                                                           ov::element::f32},
                             //  f and f
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::f32,
                                                           {1},
                                                           ov::element::f32,
                                                           true,
                                                           false,
                                                           ov::element::f32,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::f8e4m3,
                                                           {1},
                                                           ov::element::f32,
                                                           true,
                                                           false,
                                                           ov::element::f32,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::f64,
                                                           {1},
                                                           ov::element::f32,
                                                           true,
                                                           false,
                                                           ov::element::f64,
                                                           ov::element::f32},
                             //  u and i
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u1,
                                                           {1},
                                                           ov::element::i4,
                                                           true,
                                                           false,
                                                           ov::element::i4,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u16,
                                                           {1},
                                                           ov::element::i8,
                                                           true,
                                                           false,
                                                           ov::element::i32,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::i4,
                                                           {1},
                                                           ov::element::u32,
                                                           true,
                                                           false,
                                                           ov::element::i64,
                                                           ov::element::f32},
                             //  u and f
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u1,
                                                           {1},
                                                           ov::element::f8e5m2,
                                                           true,
                                                           false,
                                                           ov::element::f8e5m2,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u16,
                                                           {1},
                                                           ov::element::f32,
                                                           false,
                                                           false,
                                                           ov::element::f32,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::f16,
                                                           {1},
                                                           ov::element::u32,
                                                           true,
                                                           false,
                                                           ov::element::f16,
                                                           ov::element::f32},
                             //  i and f
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::i16,
                                                           {1},
                                                           ov::element::f32,
                                                           false,
                                                           false,
                                                           ov::element::f32,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::f16,
                                                           {1},
                                                           ov::element::i32,
                                                           true,
                                                           false,
                                                           ov::element::f16,
                                                           ov::element::f32}),
                         PrintToDummyParamName());

INSTANTIATE_TEST_SUITE_P(type_prop_pytorch_mode,
                         ConvertPromoteTypesTest,
                         testing::Values(
                             // All combinations for torch mode:
                             //  l scalar r tensor
                             ConvertPromoteTypesTestParams{{},
                                                           ov::element::i32,
                                                           ov::PartialShape().dynamic(),
                                                           ov::element::u8,
                                                           true,
                                                           true,
                                                           ov::element::dynamic,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{},
                                                           ov::element::i32,
                                                           ov::PartialShape().dynamic(),
                                                           ov::element::f16,
                                                           true,
                                                           true,
                                                           ov::element::f16,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{},
                                                           ov::element::f16,
                                                           {1},
                                                           ov::element::u32,
                                                           true,
                                                           true,
                                                           ov::element::f16,
                                                           ov::element::f32},
                             //  l tensor r scalar
                             ConvertPromoteTypesTestParams{{-1},
                                                           ov::element::i32,
                                                           {},
                                                           ov::element::u8,
                                                           true,
                                                           true,
                                                           ov::element::i32,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{{1, 5}, 3},
                                                           ov::element::f16,
                                                           {},
                                                           ov::element::u32,
                                                           true,
                                                           true,
                                                           ov::element::f16,
                                                           ov::element::f32},
                             //  l scalar r scalar
                             ConvertPromoteTypesTestParams{{},
                                                           ov::element::f16,
                                                           {},
                                                           ov::element::u64,
                                                           true,
                                                           true,
                                                           ov::element::f16,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{},
                                                           ov::element::u8,
                                                           {},
                                                           ov::element::i8,
                                                           true,
                                                           true,
                                                           ov::element::i16,
                                                           ov::element::f32},
                             //  Allowed safe mode:
                             ConvertPromoteTypesTestParams{{},
                                                           ov::element::f16,
                                                           {1},
                                                           ov::element::f32,
                                                           false,
                                                           true,
                                                           ov::element::f32,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{},
                                                           ov::element::u8,
                                                           {1},
                                                           ov::element::i16,
                                                           false,
                                                           true,
                                                           ov::element::i16,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{},
                                                           ov::element::u8,
                                                           {1},
                                                           ov::element::u16,
                                                           false,
                                                           true,
                                                           ov::element::u16,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{},
                                                           ov::element::u8,
                                                           {1},
                                                           ov::element::f16,
                                                           false,
                                                           false,
                                                           ov::element::f16,
                                                           ov::element::f32}),
                         PrintToDummyParamName());

INSTANTIATE_TEST_SUITE_P(type_prop_special_cases,
                         ConvertPromoteTypesTest,
                         testing::Values(
                             //  f8
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::f8e4m3,
                                                           {1},
                                                           ov::element::f8e5m2,
                                                           true,
                                                           false,
                                                           ov::element::f16,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::f8e4m3,
                                                           {1},
                                                           ov::element::f8e4m3,
                                                           true,
                                                           false,
                                                           ov::element::f8e4m3,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::f8e4m3,
                                                           {1},
                                                           ov::element::bf16,
                                                           true,
                                                           false,
                                                           ov::element::bf16,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::f8e4m3,
                                                           {1},
                                                           ov::element::i64,
                                                           true,
                                                           false,
                                                           ov::element::f8e4m3,
                                                           ov::element::f32},
                             //  bf16
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::bf16,
                                                           {1},
                                                           ov::element::bf16,
                                                           true,
                                                           false,
                                                           ov::element::bf16,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::f16,
                                                           {1},
                                                           ov::element::bf16,
                                                           true,
                                                           false,
                                                           ov::element::f32,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u64,
                                                           {1},
                                                           ov::element::bf16,
                                                           true,
                                                           false,
                                                           ov::element::bf16,
                                                           ov::element::f32},
                             //  u64
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u64,
                                                           {1},
                                                           ov::element::i4,
                                                           true,
                                                           false,
                                                           ov::element::f32,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u64,
                                                           {1},
                                                           ov::element::f8e4m3,
                                                           true,
                                                           false,
                                                           ov::element::f8e4m3,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u64,
                                                           {1},
                                                           ov::element::u32,
                                                           true,
                                                           false,
                                                           ov::element::u64,
                                                           ov::element::f32},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u64,
                                                           {1},
                                                           ov::element::i4,
                                                           true,
                                                           false,
                                                           ov::element::i64,
                                                           ov::element::i64},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u64,
                                                           {1},
                                                           ov::element::i4,
                                                           true,
                                                           false,
                                                           ov::element::i4,
                                                           ov::element::i4},
                             ConvertPromoteTypesTestParams{{1},
                                                           ov::element::u64,
                                                           {1},
                                                           ov::element::i4,
                                                           true,
                                                           false,
                                                           ov::element::u64,
                                                           ov::element::u64}),
                         PrintToDummyParamName());

TEST_F(ConvertPromoteTypesTest, exception_u_i_unsafe) {
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{});
    OV_EXPECT_THROW(
        std::ignore = this->make_op(in0, in1, false, false),
        ov::Exception,
        testing::HasSubstr("Unsigned integer input cannot be safely promoted into any supported signed integer."));
}

TEST_F(ConvertPromoteTypesTest, exception_u64_int_unsafe) {
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::u64, ov::Shape{});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{});
    OV_EXPECT_THROW(
        std::ignore = this->make_op(in0, in1, false, false),
        ov::Exception,
        testing::HasSubstr("Unsigned integer input cannot be safely promoted into any supported signed integer."));
}

TEST_F(ConvertPromoteTypesTest, exception_uint_float_unsafe) {
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::u16, ov::Shape{});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{});
    OV_EXPECT_THROW(std::ignore = this->make_op(in0, in1, false, false),
                    ov::Exception,
                    testing::HasSubstr("Integer input cannot be safely promoted to floating-point."));
}

TEST_F(ConvertPromoteTypesTest, exception_int_float_unsafe) {
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{});
    OV_EXPECT_THROW(std::ignore = this->make_op(in0, in1, false, false),
                    ov::Exception,
                    testing::HasSubstr("Integer input cannot be safely promoted to floating-point."));
}

TEST_F(ConvertPromoteTypesTest, exception_torch_signed_unsigned_unsafe) {
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{1});
    OV_EXPECT_THROW(std::ignore = this->make_op(in0, in1, false, true),
                    ov::Exception,
                    testing::HasSubstr("Scalar input cannot be PyTorch-like promoted using safe promotion rules."));
}

TEST_F(ConvertPromoteTypesTest, exception_torch_unsigned_unsafe) {
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::u64, ov::Shape{});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{1});
    OV_EXPECT_THROW(std::ignore = this->make_op(in0, in1, false, true),
                    ov::Exception,
                    testing::HasSubstr("Scalar input cannot be PyTorch-like promoted using safe promotion rules."));
}

TEST_F(ConvertPromoteTypesTest, exception_torch_floating_unsafe) {
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::Shape{});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    OV_EXPECT_THROW(std::ignore = this->make_op(in0, in1, false, true),
                    ov::Exception,
                    testing::HasSubstr("Scalar input cannot be PyTorch-like promoted using safe promotion rules."));
}

TEST_F(ConvertPromoteTypesTest, exception_bf16_f16_unsafe) {
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::Shape{1});
    OV_EXPECT_THROW(
        std::ignore = this->make_op(in0, in1, false, false),
        ov::Exception,
        testing::HasSubstr("Unsupported input element types for ConvertPromoteTypes with given attributes."));
}

TEST_F(ConvertPromoteTypesTest, exception_f8e4m3_f8e5m2_unsafe) {
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f8e4m3, ov::Shape{1});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f8e5m2, ov::Shape{1});
    OV_EXPECT_THROW(
        std::ignore = this->make_op(in0, in1, false, false),
        ov::Exception,
        testing::HasSubstr("Unsupported input element types for ConvertPromoteTypes with given attributes."));
}

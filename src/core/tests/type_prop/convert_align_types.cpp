// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert_align_types.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;
using namespace testing;

struct ConvertAlignTypesTestParams {
    PartialShape lhs_shape;
    element::Type lhs_type;
    PartialShape rhs_shape;
    element::Type rhs_type;
    bool pytorch_scalar_align;
    bool promote_unsafe;
    element::Type expected_type;
    element::Type u64_integer;
};
struct ConvertAlignTypesTest : ::testing::TestWithParam<ConvertAlignTypesTestParams> {};

TEST_F(ConvertAlignTypesTest, default_ctor) {
    auto lhs = std::make_shared<op::v0::Parameter>(element::u8, Shape{});
    auto rhs = std::make_shared<op::v0::Parameter>(element::f16, Shape{});
    auto c = std::make_shared<op::v14::ConvertAlignTypes>();
    c->set_arguments(OutputVector{lhs, rhs});
    c->set_pytorch_scalar_align(true);
    c->set_promote_unsafe(true);
    c->set_u64_integer_promotion_target(element::f64);
    c->validate_and_infer_types();
    ASSERT_EQ(c->get_output_element_type(0), c->get_output_element_type(1));
    ASSERT_EQ(c->get_output_element_type(0), element::f16);
    ASSERT_EQ(c->get_output_partial_shape(0), (Shape{}));
    ASSERT_EQ(c->get_output_partial_shape(1), (Shape{}));
}

TEST_P(ConvertAlignTypesTest, suite) {
    auto params = GetParam();
    auto lhs_shape = params.lhs_shape;
    auto lhs_dynamic = (lhs_shape == PartialShape().dynamic()) ? true : false;
    if (!lhs_dynamic) {
        set_shape_labels(lhs_shape, 10);
    }
    auto rhs_shape = params.rhs_shape;
    auto rhs_dynamic = (rhs_shape == PartialShape().dynamic()) ? true : false;
    if (!rhs_dynamic) {
        set_shape_labels(rhs_shape, 100);
    }

    auto lhs = std::make_shared<op::v0::Parameter>(params.lhs_type, lhs_shape);
    auto rhs = std::make_shared<op::v0::Parameter>(params.rhs_type, rhs_shape);
    auto c = std::make_shared<op::v14::ConvertAlignTypes>(lhs,
                                                          rhs,
                                                          params.pytorch_scalar_align,
                                                          params.promote_unsafe,
                                                          params.u64_integer);
    ASSERT_EQ(c->get_output_element_type(0), c->get_output_element_type(1));
    ASSERT_EQ(c->get_output_element_type(0), params.expected_type);
    ASSERT_EQ(c->get_output_partial_shape(0), (lhs_shape));
    ASSERT_EQ(c->get_output_partial_shape(1), (rhs_shape));
    ASSERT_EQ(get_shape_labels(c->get_output_partial_shape(0)), get_shape_labels(lhs_shape));
    ASSERT_EQ(get_shape_labels(c->get_output_partial_shape(1)), get_shape_labels(rhs_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    ConvertAlignTypesTest,
    ::testing::Values(
        // Test cases:
        //  dynamic
        ConvertAlignTypesTestParams{{1, 2, 3, 4},
                                    element::dynamic,
                                    {5, 6, 7},
                                    element::dynamic,
                                    false,
                                    true,
                                    element::dynamic,
                                    element::f32},
        ConvertAlignTypesTestParams{{5, 6, 7},
                                    element::dynamic,
                                    {1},
                                    element::f16,
                                    true,
                                    false,
                                    element::dynamic,
                                    element::f32},
        ConvertAlignTypesTestParams{{1, 2, 3, 4},
                                    element::i64,
                                    {1},
                                    element::dynamic,
                                    true,
                                    false,
                                    element::dynamic,
                                    element::f32},
        ConvertAlignTypesTestParams{{},
                                    element::dynamic,
                                    {1},
                                    element::f32,
                                    true,
                                    true,
                                    element::dynamic,
                                    element::f32},
        //  bool
        ConvertAlignTypesTestParams{{1},
                                    element::boolean,
                                    {1, 2, 3, 4},
                                    element::boolean,
                                    false,
                                    true,
                                    element::boolean,
                                    element::f32},
        ConvertAlignTypesTestParams{PartialShape().dynamic(),
                                    element::boolean,
                                    {-1, {1, 5}, -1, -1},
                                    element::u32,
                                    false,
                                    true,
                                    element::u32,
                                    element::f32},
        ConvertAlignTypesTestParams{{1}, element::i16, {1}, element::boolean, true, false, element::i16, element::f32},
        ConvertAlignTypesTestParams{{1}, element::f16, {1}, element::boolean, true, false, element::f16, element::f32},
        //  u and u
        ConvertAlignTypesTestParams{{1}, element::u8, {1}, element::u8, true, false, element::u8, element::f32},
        ConvertAlignTypesTestParams{{1}, element::u8, {1}, element::u4, true, false, element::u8, element::f32},
        ConvertAlignTypesTestParams{{1}, element::u1, {1}, element::u4, true, false, element::u4, element::f32},
        ConvertAlignTypesTestParams{{1}, element::u1, {1}, element::u1, true, false, element::u1, element::f32},
        //  i and i
        ConvertAlignTypesTestParams{{1}, element::i8, {1}, element::i8, true, false, element::i8, element::f32},
        ConvertAlignTypesTestParams{{1}, element::i4, {1}, element::i8, true, false, element::i8, element::f32},
        //  f and f
        ConvertAlignTypesTestParams{{1}, element::f32, {1}, element::f32, true, false, element::f32, element::f32},
        ConvertAlignTypesTestParams{{1}, element::f8e4m3, {1}, element::f32, true, false, element::f32, element::f32},
        ConvertAlignTypesTestParams{{1}, element::f64, {1}, element::f32, true, false, element::f64, element::f32},
        //  u and i
        ConvertAlignTypesTestParams{{1}, element::u1, {1}, element::i4, true, false, element::i4, element::f32},
        ConvertAlignTypesTestParams{{1}, element::u16, {1}, element::i8, true, false, element::i32, element::f32},
        ConvertAlignTypesTestParams{{1}, element::i4, {1}, element::u32, true, false, element::i64, element::f32},
        //  u and f
        ConvertAlignTypesTestParams{{1}, element::u1, {1}, element::f8e5m2, true, false, element::f8e5m2, element::f32},
        ConvertAlignTypesTestParams{{1}, element::u16, {1}, element::f32, false, false, element::f32, element::f32},
        ConvertAlignTypesTestParams{{1}, element::f16, {1}, element::u32, true, false, element::f16, element::f32},
        //  i and f
        ConvertAlignTypesTestParams{{1}, element::i16, {1}, element::f32, false, false, element::f32, element::f32},
        ConvertAlignTypesTestParams{{1}, element::f16, {1}, element::i32, true, false, element::f16, element::f32},
        // All combinations for torch mode:
        //  l scalar r tensor
        ConvertAlignTypesTestParams{{},
                                    element::i32,
                                    PartialShape().dynamic(),
                                    element::u8,
                                    true,
                                    true,
                                    element::u8,
                                    element::f32},
        ConvertAlignTypesTestParams{{}, element::f16, {1}, element::u32, true, true, element::f16, element::f32},
        //  l tensor r scalar
        ConvertAlignTypesTestParams{{-1}, element::i32, {}, element::u8, true, true, element::i32, element::f32},
        ConvertAlignTypesTestParams{{{1, 5}, 3},
                                    element::f16,
                                    {},
                                    element::u32,
                                    true,
                                    true,
                                    element::f16,
                                    element::f32},
        //  l scalar r scalar
        ConvertAlignTypesTestParams{{}, element::f16, {}, element::u64, true, true, element::f16, element::f32},
        ConvertAlignTypesTestParams{{}, element::u8, {}, element::i8, true, true, element::i16, element::f32},
        //  Allowed safe mode:
        ConvertAlignTypesTestParams{{}, element::f16, {1}, element::f32, false, true, element::f32, element::f32},
        ConvertAlignTypesTestParams{{}, element::u8, {1}, element::i16, false, true, element::i16, element::f32},
        ConvertAlignTypesTestParams{{}, element::u8, {1}, element::u16, false, true, element::u16, element::f32},
        ConvertAlignTypesTestParams{{}, element::u8, {1}, element::f16, false, false, element::f16, element::f32},
        // Special cases:
        //  f8
        ConvertAlignTypesTestParams{{1},
                                    element::f8e4m3,
                                    {1},
                                    element::f8e5m2,
                                    true,
                                    false,
                                    element::f16,
                                    element::f32},
        ConvertAlignTypesTestParams{{1},
                                    element::f8e4m3,
                                    {1},
                                    element::f8e4m3,
                                    true,
                                    false,
                                    element::f8e4m3,
                                    element::f32},
        ConvertAlignTypesTestParams{{1}, element::f8e4m3, {1}, element::bf16, true, false, element::bf16, element::f32},
        ConvertAlignTypesTestParams{{1},
                                    element::f8e4m3,
                                    {1},
                                    element::i64,
                                    true,
                                    false,
                                    element::f8e4m3,
                                    element::f32},
        //  bf16
        ConvertAlignTypesTestParams{{1}, element::bf16, {1}, element::bf16, true, false, element::bf16, element::f32},
        ConvertAlignTypesTestParams{{1}, element::f16, {1}, element::bf16, true, false, element::f32, element::f32},
        ConvertAlignTypesTestParams{{1}, element::u64, {1}, element::bf16, true, false, element::bf16, element::f32},
        //  u64
        ConvertAlignTypesTestParams{{1}, element::u64, {1}, element::i4, true, false, element::f32, element::f32},
        ConvertAlignTypesTestParams{{1},
                                    element::u64,
                                    {1},
                                    element::f8e4m3,
                                    true,
                                    false,
                                    element::f8e4m3,
                                    element::f32},
        ConvertAlignTypesTestParams{{1}, element::u64, {1}, element::u32, true, false, element::u64, element::f32},
        ConvertAlignTypesTestParams{{1}, element::u64, {1}, element::i4, true, false, element::i64, element::i64},
        ConvertAlignTypesTestParams{{1}, element::u64, {1}, element::i4, true, false, element::i4, element::i4},
        ConvertAlignTypesTestParams{{1}, element::u64, {1}, element::i4, true, false, element::u64, element::u64}),
    PrintToDummyParamName());

TEST_F(ConvertAlignTypesTest, exception_u_i_unsafe) {
    auto lhs = std::make_shared<op::v0::Parameter>(element::u8, Shape{});
    auto rhs = std::make_shared<op::v0::Parameter>(element::i8, Shape{});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::ConvertAlignTypes>(lhs, rhs, false, false),
                    Exception,
                    HasSubstr("Unsigned integer input cannot be safely promoted into any supported signed integer."));
}

TEST_F(ConvertAlignTypesTest, exception_u64_int_unsafe) {
    auto lhs = std::make_shared<op::v0::Parameter>(element::u64, Shape{});
    auto rhs = std::make_shared<op::v0::Parameter>(element::i8, Shape{});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::ConvertAlignTypes>(lhs, rhs, false, false),
                    Exception,
                    HasSubstr("Unsigned integer input cannot be safely promoted into any supported signed integer."));
}

TEST_F(ConvertAlignTypesTest, exception_uint_float_unsafe) {
    auto lhs = std::make_shared<op::v0::Parameter>(element::u16, Shape{});
    auto rhs = std::make_shared<op::v0::Parameter>(element::f16, Shape{});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::ConvertAlignTypes>(lhs, rhs, false, false),
                    Exception,
                    HasSubstr("Integer input cannot be safely promoted to floating-point."));
}

TEST_F(ConvertAlignTypesTest, exception_int_float_unsafe) {
    auto lhs = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto rhs = std::make_shared<op::v0::Parameter>(element::f16, Shape{});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::ConvertAlignTypes>(lhs, rhs, false, false),
                    Exception,
                    HasSubstr("Integer input cannot be safely promoted to floating-point."));
}

TEST_F(ConvertAlignTypesTest, exception_torch_signed_unsigned_unsafe) {
    auto lhs = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
    auto rhs = std::make_shared<op::v0::Parameter>(element::u4, Shape{1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::ConvertAlignTypes>(lhs, rhs, false, true),
                    Exception,
                    HasSubstr("Scalar input cannot be PyTorch-like aligned using safe promotion rules."));
}

TEST_F(ConvertAlignTypesTest, exception_torch_unsigned_unsafe) {
    auto lhs = std::make_shared<op::v0::Parameter>(element::u64, Shape{});
    auto rhs = std::make_shared<op::v0::Parameter>(element::u4, Shape{1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::ConvertAlignTypes>(lhs, rhs, false, true),
                    Exception,
                    HasSubstr("Scalar input cannot be PyTorch-like aligned using safe promotion rules."));
}

TEST_F(ConvertAlignTypesTest, exception_torch_floating_unsafe) {
    auto lhs = std::make_shared<op::v0::Parameter>(element::f64, Shape{});
    auto rhs = std::make_shared<op::v0::Parameter>(element::f16, Shape{1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::ConvertAlignTypes>(lhs, rhs, false, true),
                    Exception,
                    HasSubstr("Scalar input cannot be PyTorch-like aligned using safe promotion rules."));
}

TEST_F(ConvertAlignTypesTest, exception_bf16_f16_unsafe) {
    auto lhs = std::make_shared<op::v0::Parameter>(element::f16, Shape{1});
    auto rhs = std::make_shared<op::v0::Parameter>(element::bf16, Shape{1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::ConvertAlignTypes>(lhs, rhs, false, false),
                    Exception,
                    HasSubstr("Unsupported input element types for ConvertAlignTypes with given attributes."));
}

TEST_F(ConvertAlignTypesTest, exception_f8e4m3_f8e5m2_unsafe) {
    auto lhs = std::make_shared<op::v0::Parameter>(element::f8e4m3, Shape{1});
    auto rhs = std::make_shared<op::v0::Parameter>(element::f8e5m2, Shape{1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::ConvertAlignTypes>(lhs, rhs, false, false),
                    Exception,
                    HasSubstr("Unsupported input element types for ConvertAlignTypes with given attributes."));
}

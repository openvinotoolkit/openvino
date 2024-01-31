// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert_align_types.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, convert_align_types_op_default) {
    NodeBuilder::opset().insert<op::v1::ConvertAlignTypes>();
    auto lhs = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 4});
    auto rhs = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 4});

    const auto convert = std::make_shared<op::v1::ConvertAlignTypes>(lhs, rhs);
    NodeBuilder builder(convert, {lhs, rhs});

    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    const auto g_convert = ov::as_type_ptr<op::v1::ConvertAlignTypes>(builder.create());
    EXPECT_EQ(g_convert->get_pytorch_scalar_align(), false);
    EXPECT_EQ(g_convert->get_promote_unsafe(), false);
    EXPECT_EQ(g_convert->get_u64_integer_promotion_target(), element::f32);
}

TEST(attributes, convert_align_types_constructor) {
    NodeBuilder::opset().insert<op::v1::ConvertAlignTypes>();
    auto lhs = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 4});
    auto rhs = std::make_shared<op::v0::Parameter>(element::i32, Shape{2, 4});
    const bool pytorch_scalar_align = true;
    const bool promote_unsafe = true;

    const auto convert =
        std::make_shared<op::v1::ConvertAlignTypes>(lhs, rhs, pytorch_scalar_align, promote_unsafe, element::i64);
    NodeBuilder builder(convert, {lhs, rhs});

    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    const auto g_convert = ov::as_type_ptr<op::v1::ConvertAlignTypes>(builder.create());
    EXPECT_EQ(g_convert->get_pytorch_scalar_align(), convert->get_pytorch_scalar_align());
    EXPECT_EQ(g_convert->get_promote_unsafe(), convert->get_promote_unsafe());
    EXPECT_EQ(g_convert->get_u64_integer_promotion_target(), element::i64);
}

TEST(attributes, convert_align_types_op_setters) {
    NodeBuilder::opset().insert<op::v1::ConvertAlignTypes>();
    auto lhs = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 4});
    auto rhs = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 4});

    const auto convert = std::make_shared<op::v1::ConvertAlignTypes>(lhs, rhs);
    convert->set_promote_unsafe(true);
    convert->set_pytorch_scalar_align(true);
    convert->set_u64_integer_promotion_target(element::u64);
    NodeBuilder builder(convert, {lhs, rhs});

    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    const auto g_convert = ov::as_type_ptr<op::v1::ConvertAlignTypes>(builder.create());
    EXPECT_EQ(g_convert->get_pytorch_scalar_align(), true);
    EXPECT_EQ(g_convert->get_promote_unsafe(), true);
    EXPECT_EQ(g_convert->get_u64_integer_promotion_target(), element::u64);
}

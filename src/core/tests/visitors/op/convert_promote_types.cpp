// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert_promote_types.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::test::NodeBuilder;

TEST(attributes, convert_promote_types_op_default) {
    NodeBuilder::opset().insert<ov::op::v14::ConvertPromoteTypes>();
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 4});

    const auto convert = std::make_shared<ov::op::v14::ConvertPromoteTypes>(in0, in1);
    NodeBuilder builder(convert, {in0, in1});

    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    const auto g_convert = ov::as_type_ptr<ov::op::v14::ConvertPromoteTypes>(builder.create());
    EXPECT_EQ(g_convert->get_pytorch_scalar_promotion(), false);
    EXPECT_EQ(g_convert->get_promote_unsafe(), false);
    EXPECT_EQ(g_convert->get_u64_integer_promotion_target(), ov::element::f32);
}

TEST(attributes, convert_promote_types_constructor) {
    NodeBuilder::opset().insert<ov::op::v14::ConvertPromoteTypes>();
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 4});
    const bool pytorch_scalar_promotion = true;
    const bool promote_unsafe = true;

    const auto convert = std::make_shared<ov::op::v14::ConvertPromoteTypes>(in0,
                                                                            in1,
                                                                            pytorch_scalar_promotion,
                                                                            promote_unsafe,
                                                                            ov::element::i64);
    NodeBuilder builder(convert, {in0, in1});

    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    const auto g_convert = ov::as_type_ptr<ov::op::v14::ConvertPromoteTypes>(builder.create());
    EXPECT_EQ(g_convert->get_pytorch_scalar_promotion(), convert->get_pytorch_scalar_promotion());
    EXPECT_EQ(g_convert->get_promote_unsafe(), convert->get_promote_unsafe());
    EXPECT_EQ(g_convert->get_u64_integer_promotion_target(), ov::element::i64);
}

TEST(attributes, convert_promote_types_op_setters) {
    NodeBuilder::opset().insert<ov::op::v14::ConvertPromoteTypes>();
    auto in0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 4});

    const auto convert = std::make_shared<ov::op::v14::ConvertPromoteTypes>(in0, in1);
    convert->set_promote_unsafe(true);
    convert->set_pytorch_scalar_promotion(true);
    convert->set_u64_integer_promotion_target(ov::element::u64);
    NodeBuilder builder(convert, {in0, in1});

    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    const auto g_convert = ov::as_type_ptr<ov::op::v14::ConvertPromoteTypes>(builder.create());
    EXPECT_EQ(g_convert->get_pytorch_scalar_promotion(), true);
    EXPECT_EQ(g_convert->get_promote_unsafe(), true);
    EXPECT_EQ(g_convert->get_u64_integer_promotion_target(), ov::element::u64);
}

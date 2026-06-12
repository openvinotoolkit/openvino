// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bevpool_v2.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

namespace ov {
namespace test {

TEST(attributes, bevpool_v2) {
    NodeBuilder::opset().insert<ov::op::v15::BevPoolV2>();

    const auto cf = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 4, 3, 5});
    const auto dw = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, 3, 5});
    const auto idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{6});
    const auto itv = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2, 3});

    const ov::op::v15::Bound x_bound{-51.2f, 51.2f, 0.8f};
    const ov::op::v15::Bound y_bound{-51.2f, 51.2f, 0.8f};
    const ov::op::v15::Bound z_bound{-5.0f, 3.0f, 8.0f};
    const ov::op::v15::Bound d_bound{1.0f, 60.0f, 1.0f};

    const auto op = std::make_shared<ov::op::v15::BevPoolV2>(ov::OutputVector{cf, dw, idx, itv},
                                                              4,
                                                              8,
                                                              5,
                                                              3,
                                                              128,
                                                              128,
                                                              x_bound,
                                                              y_bound,
                                                              z_bound,
                                                              d_bound);

    NodeBuilder builder(op, {cf, dw, idx, itv});
    const auto g_op = ov::as_type_ptr<ov::op::v15::BevPoolV2>(builder.create());

    constexpr auto expected_attr_count = 18;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(op->get_input_channels(), g_op->get_input_channels());
    EXPECT_EQ(op->get_output_channels(), g_op->get_output_channels());
    EXPECT_EQ(op->get_image_width(), g_op->get_image_width());
    EXPECT_EQ(op->get_image_height(), g_op->get_image_height());
    EXPECT_EQ(op->get_feature_width(), g_op->get_feature_width());
    EXPECT_EQ(op->get_feature_height(), g_op->get_feature_height());

    EXPECT_FLOAT_EQ(op->get_x_bound().min, g_op->get_x_bound().min);
    EXPECT_FLOAT_EQ(op->get_x_bound().max, g_op->get_x_bound().max);
    EXPECT_FLOAT_EQ(op->get_x_bound().step, g_op->get_x_bound().step);

    EXPECT_FLOAT_EQ(op->get_y_bound().min, g_op->get_y_bound().min);
    EXPECT_FLOAT_EQ(op->get_y_bound().max, g_op->get_y_bound().max);
    EXPECT_FLOAT_EQ(op->get_y_bound().step, g_op->get_y_bound().step);

    EXPECT_FLOAT_EQ(op->get_z_bound().min, g_op->get_z_bound().min);
    EXPECT_FLOAT_EQ(op->get_z_bound().max, g_op->get_z_bound().max);
    EXPECT_FLOAT_EQ(op->get_z_bound().step, g_op->get_z_bound().step);

    EXPECT_FLOAT_EQ(op->get_d_bound().min, g_op->get_d_bound().min);
    EXPECT_FLOAT_EQ(op->get_d_bound().max, g_op->get_d_bound().max);
    EXPECT_FLOAT_EQ(op->get_d_bound().step, g_op->get_d_bound().step);
}

}  // namespace test
}  // namespace ov

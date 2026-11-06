// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_residual.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {

TEST(attributes, gated_residual_read_default_attrs) {
    NodeBuilder::opset().insert<op::internal::GatedResidual>();
    const auto residual = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3, 16});
    const auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{16});
    const auto w_down = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{8, 16});
    const auto w_up = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{16, 8});

    const auto op = std::make_shared<op::internal::GatedResidual>(OutputVector{residual, norm_weight, w_down, w_up},
                                                                  op::internal::GatedResidual::Mode::READ,
                                                                  4,
                                                                  4,
                                                                  8,
                                                                  1e-6F,
                                                                  true,
                                                                  false);

    NodeBuilder builder(op, {residual, norm_weight, w_down, w_up});
    auto g_op = ov::as_type_ptr<op::internal::GatedResidual>(builder.create());

    constexpr auto expected_attr_count = 7;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_op->get_mode(), op->get_mode());
    EXPECT_EQ(g_op->get_hc_count(), op->get_hc_count());
    EXPECT_EQ(g_op->get_hidden_size(), op->get_hidden_size());
    EXPECT_EQ(g_op->get_lowrank(), op->get_lowrank());
    EXPECT_FLOAT_EQ(g_op->get_epsilon(), op->get_epsilon());
    EXPECT_EQ(g_op->get_norm_gain_plus_one(), op->get_norm_gain_plus_one());
    EXPECT_EQ(g_op->get_produce_write_gate(), op->get_produce_write_gate());
}

TEST(attributes, gated_residual_read_compressed_weights_attrs) {
    NodeBuilder::opset().insert<op::internal::GatedResidual>();
    // The compressed form is not an attribute: it follows from the weight types and the input count.
    const auto residual = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3, 16});
    const auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{16});
    const auto w_down = std::make_shared<op::v0::Parameter>(element::u4, PartialShape{8, 16});
    const auto w_up = std::make_shared<op::v0::Parameter>(element::u4, PartialShape{16, 8});
    const auto w_down_scale = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{2, 8});
    const auto w_up_scale = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{2, 16});
    const auto w_down_zp = std::make_shared<op::v0::Parameter>(element::u4, PartialShape{2, 8});
    const auto w_up_zp = std::make_shared<op::v0::Parameter>(element::u4, PartialShape{2, 16});

    const OutputVector args{residual, norm_weight, w_down, w_up, w_down_scale, w_up_scale, w_down_zp, w_up_zp};
    const auto op = std::make_shared<op::internal::GatedResidual>(args,
                                                                  op::internal::GatedResidual::Mode::READ,
                                                                  4,
                                                                  4,
                                                                  8,
                                                                  1e-6F,
                                                                  true,
                                                                  false);

    NodeBuilder builder(op, args);
    auto g_op = ov::as_type_ptr<op::internal::GatedResidual>(builder.create());

    constexpr auto expected_attr_count = 7;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    const auto layout = g_op->get_read_layout();
    ASSERT_TRUE(layout.has_value());
    EXPECT_TRUE(layout->compressed);
    EXPECT_TRUE(layout->has_zp);
}

TEST(attributes, gated_residual_write_non_default_attrs) {
    NodeBuilder::opset().insert<op::internal::GatedResidual>();
    const auto residual = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{2, 3, 16});
    const auto sublayer_y = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{2, 3, 4});
    const auto write_gate = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{2, 3, 4});

    const auto op = std::make_shared<op::internal::GatedResidual>(OutputVector{residual, sublayer_y, write_gate},
                                                                  op::internal::GatedResidual::Mode::WRITE,
                                                                  4,
                                                                  4,
                                                                  0,
                                                                  1e-3F,
                                                                  false,
                                                                  false);

    NodeBuilder builder(op, {residual, sublayer_y, write_gate});
    auto g_op = ov::as_type_ptr<op::internal::GatedResidual>(builder.create());

    constexpr auto expected_attr_count = 7;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_op->get_mode(), op::internal::GatedResidual::Mode::WRITE);
    EXPECT_EQ(g_op->get_hc_count(), 4U);
    EXPECT_EQ(g_op->get_hidden_size(), 4U);
    EXPECT_EQ(g_op->get_lowrank(), 0U);
    EXPECT_FLOAT_EQ(g_op->get_epsilon(), 1e-3F);
    EXPECT_FALSE(g_op->get_norm_gain_plus_one());
    EXPECT_FALSE(g_op->get_produce_write_gate());

    EXPECT_EQ(g_op->get_mode(), op->get_mode());
    EXPECT_EQ(g_op->get_hc_count(), op->get_hc_count());
    EXPECT_EQ(g_op->get_hidden_size(), op->get_hidden_size());
    EXPECT_EQ(g_op->get_lowrank(), op->get_lowrank());
    EXPECT_FLOAT_EQ(g_op->get_epsilon(), op->get_epsilon());
    EXPECT_EQ(g_op->get_norm_gain_plus_one(), op->get_norm_gain_plus_one());
    EXPECT_EQ(g_op->get_produce_write_gate(), op->get_produce_write_gate());
}

}  // namespace ov::test

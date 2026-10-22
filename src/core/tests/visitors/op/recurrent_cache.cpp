// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/recurrent_cache.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {
namespace {
OutputVector cache_inputs() {
    OutputVector inputs;
    for (const auto& shape : std::vector<PartialShape>{{1, 1, 12, 35},
                                                       {1, 1, 1, -1},
                                                       {1, 1, 1, -1},
                                                       {1, 1, 1, 1},
                                                       {1, 1, 1, 1},
                                                       {1, 1, 1, 2}}) {
        inputs.push_back(std::make_shared<op::v0::Parameter>(inputs.empty() ? element::f32 : element::i64, shape));
    }
    return inputs;
}
}  // namespace

TEST(attributes, recurrent_state_gather) {
    NodeBuilder::opset().insert<op::internal::RecurrentStateGather>();
    const auto inputs = cache_inputs();
    const auto node = std::make_shared<op::internal::RecurrentStateGather>(inputs, 5, 7);
    NodeBuilder builder(node, inputs);
    const auto restored = as_type_ptr<op::internal::RecurrentStateGather>(builder.create());
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(builder.get_value_map_size(), 3);
    EXPECT_EQ(restored->get_key_size(), 5);
    EXPECT_EQ(restored->get_value_size(), 7);
    EXPECT_EQ(restored->get_output_partial_shape(0), node->get_output_partial_shape(0));
    EXPECT_EQ(restored->get_output_partial_shape(1), node->get_output_partial_shape(1));
}

TEST(attributes, recurrent_state_update) {
    NodeBuilder::opset().insert<op::internal::RecurrentStateUpdate>();
    const OutputVector inputs{std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, 12, 35}),
                              std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 1, -1, 35}),
                              std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, 1, 1, 4})};
    const auto node = std::make_shared<op::internal::RecurrentStateUpdate>(inputs, 5, 7);
    NodeBuilder builder(node, inputs);
    const auto restored = as_type_ptr<op::internal::RecurrentStateUpdate>(builder.create());
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(builder.get_value_map_size(), 2);
    EXPECT_EQ(restored->get_key_size(), 5);
    EXPECT_EQ(restored->get_value_size(), 7);
    EXPECT_EQ(restored->get_output_partial_shape(0), node->get_output_partial_shape(0));
}

TEST(attributes, recurrent_conv_cache) {
    NodeBuilder::opset().insert<op::internal::RecurrentConvCache>();
    auto inputs = cache_inputs();
    inputs[0] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, 12, 21});
    inputs.push_back(std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, -1, 7, -1}));
    inputs.push_back(std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, 1, -1, 2}));
    const auto node = std::make_shared<op::internal::RecurrentConvCache>(inputs, 3);
    NodeBuilder builder(node, inputs);
    const auto restored = as_type_ptr<op::internal::RecurrentConvCache>(builder.create());
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(builder.get_value_map_size(), 1);
    EXPECT_EQ(restored->get_history(), 3);
    EXPECT_EQ(restored->get_output_partial_shape(0), node->get_output_partial_shape(0));
    EXPECT_EQ(restored->get_output_partial_shape(1), node->get_output_partial_shape(1));
}
}  // namespace ov::test

namespace ov::test {
TEST(attributes, recurrent_active_only) {
    NodeBuilder::opset().insert<op::internal::RecurrentStateGather>();
    auto inputs = cache_inputs();
    auto node = std::make_shared<op::internal::RecurrentStateGather>(inputs, 5, 7, false);
    NodeBuilder builder(node, inputs);
    auto restored = as_type_ptr<op::internal::RecurrentStateGather>(builder.create());
    ASSERT_NE(restored, nullptr);
    EXPECT_FALSE(restored->get_prepare_cache());
    EXPECT_EQ(restored->get_output_size(), 1);
}
TEST(attributes, recurrent_gdn) {
    NodeBuilder::opset().insert<op::internal::RecurrentGatedDeltaNet>();
    auto param = [](element::Type type, PartialShape shape) {
        return std::make_shared<op::v0::Parameter>(type, shape);
    };
    OutputVector inputs{param(element::f16, {-1, -1, 2, 128}),
                        param(element::f16, {-1, -1, 2, 128}),
                        param(element::f16, {-1, -1, 2, 8}),
                        param(element::f32, {1, 1, 12, 2048}),
                        param(element::f16, {-1, -1, 2}),
                        param(element::f16, {-1, -1, 2}),
                        param(element::i32, {1, 1, 1, -1}),
                        param(element::i64, {1, 1, 1, -1}),
                        param(element::i32, {1, 1, 1, 1}),
                        param(element::i64, {1, 1, 1, 1}),
                        param(element::i32, {1, 1, 1, 2}),
                        param(element::i64, {1, 1, 1, 4})};
    auto node = std::make_shared<op::internal::RecurrentGatedDeltaNet>(inputs, true, 1e-5f, 2e-5f, 3);
    NodeBuilder builder(node, inputs);
    auto restored = as_type_ptr<op::internal::RecurrentGatedDeltaNet>(builder.create());
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(builder.get_value_map_size(), 4);
    EXPECT_EQ(restored->get_snapshot_count(), 3);
    EXPECT_EQ(restored->get_q_l2_norm_eps(), 1e-5f);
    EXPECT_EQ(restored->get_k_l2_norm_eps(), 2e-5f);
    EXPECT_TRUE(restored->get_fuse_qk_l2norm());
    EXPECT_EQ(restored->get_output_partial_shape(1), node->get_output_partial_shape(1));
}
}  // namespace ov::test

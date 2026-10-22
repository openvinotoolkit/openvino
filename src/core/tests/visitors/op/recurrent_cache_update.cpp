// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "openvino/op/recurrent_cache_update.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"
namespace {
ov::OutputVector conv_inputs() {
    using namespace ov;
    OutputVector args;
    for (const auto& s : std::vector<PartialShape>{{1, 1, 12, 21},
                                                   {1, 1, 1, -1},
                                                   {1, 1, 1, -1},
                                                   {1, 1, 1, 1},
                                                   {1, 1, 1, 1},
                                                   {1, 1, 1, 2}})
        args.push_back(std::make_shared<op::v0::Parameter>(args.empty() ? element::f32 : element::i64, s));
    args.push_back(std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, -1, 7, -1}));
    args.push_back(std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, 1, -1, 2}));
    args.push_back(std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 1, 7, 4}));
    return args;
}
ov::OutputVector gdn_inputs() {
    using namespace ov;
    auto p = [](element::Type t, PartialShape s) {
        return std::make_shared<op::v0::Parameter>(t, s);
    };
    return {p(element::f16, {-1, -1, 2, 128}),
            p(element::f16, {-1, -1, 2, 128}),
            p(element::f16, {-1, -1, 4, 8}),
            p(element::f32, {1, 1, 12, 4096}),
            p(element::f16, {-1, -1, 4}),
            p(element::f16, {-1, -1, 4}),
            p(element::i32, {1, 1, 1, -1}),
            p(element::i64, {1, 1, 1, -1}),
            p(element::i32, {1, 1, 1, 1}),
            p(element::i64, {1, 1, 1, 1}),
            p(element::i32, {1, 1, 1, 2}),
            p(element::i64, {1, 1, 1, 4})};
}
}  // namespace

namespace ov::test {
TEST(attributes, recurrent_conv_cache_update) {
    NodeBuilder::opset().insert<op::internal::RecurrentConvCacheUpdate>();
    auto args = conv_inputs();
    auto node = std::make_shared<op::internal::RecurrentConvCacheUpdate>(args, 3);
    NodeBuilder builder(node, args);
    auto restored = as_type_ptr<op::internal::RecurrentConvCacheUpdate>(builder.create());
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(builder.get_value_map_size(), 1);
    EXPECT_EQ(restored->get_history(), 3);
    EXPECT_EQ(restored->get_output_partial_shape(1), node->get_output_partial_shape(1));
}
TEST(attributes, recurrent_gdn_update) {
    NodeBuilder::opset().insert<op::internal::RecurrentGatedDeltaNetUpdate>();
    auto args = gdn_inputs();
    auto node = std::make_shared<op::internal::RecurrentGatedDeltaNetUpdate>(args, true, 1e-5f, 2e-5f, 3);
    NodeBuilder builder(node, args);
    auto restored = as_type_ptr<op::internal::RecurrentGatedDeltaNetUpdate>(builder.create());
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(builder.get_value_map_size(), 4);
    EXPECT_TRUE(restored->get_fuse_qk_l2norm());
    EXPECT_FLOAT_EQ(restored->get_q_l2_norm_eps(), 1e-5f);
    EXPECT_FLOAT_EQ(restored->get_k_l2_norm_eps(), 2e-5f);
    EXPECT_EQ(restored->get_snapshot_count(), 3);
    EXPECT_EQ(restored->get_output_partial_shape(1), node->get_output_partial_shape(1));
}
}  // namespace ov::test

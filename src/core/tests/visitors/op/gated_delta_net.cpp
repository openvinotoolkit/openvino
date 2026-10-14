// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_delta_net.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {

TEST(attributes, gated_delta_net_snapshots) {
    NodeBuilder::opset().insert<op::internal::GatedDeltaNet>();
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, 2, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8, 16});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, 4});
    const OutputVector inputs{query, query, value, state, gate, gate};
    for (const int64_t count : {1, 3}) {
        const auto op = std::make_shared<op::internal::GatedDeltaNet>(inputs, true, 1e-5f, 1e-4f, count);
        NodeBuilder builder(op, inputs);
        const auto restored = as_type_ptr<op::internal::GatedDeltaNet>(builder.create());
        ASSERT_NE(restored, nullptr);
        EXPECT_EQ(builder.get_value_map_size(), 4);
        EXPECT_TRUE(restored->get_fuse_qk_l2norm());
        EXPECT_FLOAT_EQ(restored->get_q_l2_norm_eps(), 1e-5f);
        EXPECT_FLOAT_EQ(restored->get_k_l2_norm_eps(), 1e-4f);
        EXPECT_EQ(restored->get_snapshot_count(), count);
        EXPECT_EQ(restored->get_output_size(), op->get_output_size());
    }
}

}  // namespace ov::test

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_transformations/rank_weights.hpp"

#include <gtest/gtest.h>

#include "l1_test_utils.hpp"
#include "tp_test_models.hpp"

namespace ov::tp::tests {
namespace {

/// The constants' bytes the way rank_weights counts them: f32 at two bytes per element.
size_t device_bytes(const ov::Model& model) {
    size_t bytes = 0;
    for (const auto& node : model.get_ops()) {
        if (const auto c = ov::as_type_ptr<ov::op::v0::Constant>(node))
            bytes += c->get_element_type() == ov::element::f32 ? ov::shape_size(c->get_shape()) * 2 : c->get_byte_size();
    }
    return bytes;
}

// Read off the analyzed model, the estimate is what each rank model holds once it is built.
TEST(TPRankWeightsTest, MatchTheRankModels) {
    for (const size_t world_size : {2u, 4u}) {
        SCOPED_TRACE(world_size);
        auto analyzed = make_block_with_lm_head(BlockConfig{}, 768);
        analyze(analyzed, world_size);
        const auto weights = rank_weights(*analyzed, world_size);
        ASSERT_EQ(weights.per_rank.size(), world_size);
        EXPECT_GT(weights.split, weights.replicated);
        for (size_t rank = 0; rank < world_size; ++rank) {
            const auto model = rank_model(analyzed, rank, world_size);
            // Shape constants of the rank model differ in value, never in size; the collectives add none.
            EXPECT_EQ(weights.per_rank[rank], device_bytes(*model)) << "rank " << rank;
        }
    }
}

TEST(TPRankWeightsTest, MinWorldSize) {
    RankWeights weights;
    weights.replicated = 2;
    weights.split = 10;
    EXPECT_EQ(min_world_size(weights, 12), 1u);
    EXPECT_EQ(min_world_size(weights, 7), 2u);
    EXPECT_EQ(min_world_size(weights, 5), 4u);  // 2 + ceil(10 / 4) = 5
    EXPECT_EQ(min_world_size(weights, 2), 0u);  // the whole part alone fills the device
}

}  // namespace
}  // namespace ov::tp::tests

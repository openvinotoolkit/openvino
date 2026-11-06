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
    BlockConfig config;
    config.intermediate = 512;  // whole quantization groups on 2 and 4 ranks: no runtime slices
    for (const size_t world_size : {2u, 4u}) {
        SCOPED_TRACE(world_size);
        auto analyzed = make_block_with_lm_head(config, 768);
        const auto report = analyze(analyzed, world_size);
        ASSERT_TRUE(report->runtime_slices.empty());
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

// A weight cut by a Slice at run time is whole in the rank model but folded to the rank's share on
// the device; the estimate counts the share of every constant behind the sliced input.
TEST(TPRankWeightsTest, CountTheShareOfARuntimeSlicedWeight) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 16});
    auto weight = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{8, 16}, std::vector<float>(128, 1.0f));
    auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(weight, order);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(x, transposed);
    auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{8}, std::vector<float>(8, 0.0f));
    auto model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v1::Add>(matmul, bias)},
                                             ov::ParameterVector{x});
    ShardAnnotation slice(ShardStrategy::COLUMN_PARALLEL, 1, {Segment{8, 1}}, "test");
    slice.input = 1;
    set_shard(matmul, slice);

    const auto weights = rank_weights(*model, 4);
    EXPECT_EQ(weights.replicated, bias->get_byte_size());
    EXPECT_EQ(weights.split, weight->get_byte_size() + order->get_byte_size());
    for (size_t rank = 0; rank < 4; ++rank)
        EXPECT_EQ(weights.per_rank[rank], weights.replicated + weights.split / 4) << "rank " << rank;
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

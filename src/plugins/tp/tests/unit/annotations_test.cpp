// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "tp_transformations/annotations.hpp"

namespace ov::tp::tests {
namespace {

using ::testing::HasSubstr;

TEST(TPAnnotationsTest, RangesSplitWholeUnitsWithTheRemainderOnTheFirstRanks) {
    // 8 KV heads x 128 over 3 ranks: 3, 3, 2 heads.
    const std::vector<Segment> heads{{1024, 128}};
    EXPECT_EQ(rank_ranges(heads, 0, 3), (std::vector<Range>{{0, 384}}));
    EXPECT_EQ(rank_ranges(heads, 1, 3), (std::vector<Range>{{384, 384}}));
    EXPECT_EQ(rank_ranges(heads, 2, 3), (std::vector<Range>{{768, 256}}));
}

TEST(TPAnnotationsTest, RangesOfFusedSegmentsAreOffsetBySegment) {
    // Q 8 heads, K 2 heads, V 2 heads of 4; rank 1 of 2.
    const std::vector<Segment> qkv{{32, 16}, {8, 4}, {8, 4}};
    EXPECT_EQ(rank_ranges(qkv, 1, 2), (std::vector<Range>{{16, 16}, {36, 4}, {44, 4}}));
    EXPECT_EQ(rank_length(qkv, 1, 2), 24u);
}

TEST(TPAnnotationsTest, RangesRejectASegmentThatIsNotWholeUnits) {
    OV_EXPECT_THROW(rank_ranges({{100, 32}}, 0, 2), ov::Exception, HasSubstr("in units of 32"));
}

TEST(TPAnnotationsTest, AnnotationsSurviveCloneAndConflictsAreRejected) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    auto c = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, {1, 2, 3, 4});
    auto add = std::make_shared<ov::op::v1::Add>(x, c);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{x});

    const ShardAnnotation shard(ShardStrategy::COLUMN_PARALLEL, 0, {{4, 2}}, "test");
    set_shard(c, shard);
    set_shard(c, shard);  // the same claim again is fine
    set_role(add, ProjectionRole(ShardStrategy::ROW_PARALLEL, 0, "test"));
    OV_EXPECT_THROW(set_shard(c, ShardAnnotation(ShardStrategy::ROW_PARALLEL, 0, {{4, 1}}, "other")),
                    ov::Exception,
                    HasSubstr("claimed twice"));

    const auto clone = model->clone();
    for (const auto& node : clone->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            ASSERT_NE(get_shard(*node), nullptr);
            EXPECT_EQ(*get_shard(*node), shard);
        }
        if (ov::is_type<ov::op::v1::Add>(node)) {
            ASSERT_NE(get_role(*node), nullptr);
            EXPECT_EQ(get_role(*node)->allreduce_group_id, 0);
        }
    }

    strip_annotations(*clone);
    for (const auto& node : clone->get_ordered_ops()) {
        EXPECT_EQ(get_shard(*node), nullptr);
        EXPECT_EQ(get_role(*node), nullptr);
    }
    EXPECT_NE(get_shard(*c), nullptr) << "stripping a clone must not touch the original";
}

}  // namespace
}  // namespace ov::tp::tests

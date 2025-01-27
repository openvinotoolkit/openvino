// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/segment_max.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::op::v0::Parameter;
using ov::op::v16::SegmentMax;
using ov::test::NodeBuilder;

TEST(attributes, segment_max_v16_with_num_segments) {
    NodeBuilder::opset().insert<SegmentMax>();
    const auto data = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(ov::element::i64, ov::Shape{5});
    const auto num_segments = std::make_shared<Parameter>(ov::element::i64, ov::Shape{});

    const auto op = std::make_shared<SegmentMax>(data, segment_ids, num_segments, 0);
    NodeBuilder builder(op, {data, segment_ids, num_segments});
    auto g_op = ov::as_type_ptr<SegmentMax>(builder.create());

    EXPECT_EQ(g_op->get_empty_segment_value(), op->get_empty_segment_value());
}

TEST(attributes, segment_max_v16_without_num_segments) {
    NodeBuilder::opset().insert<SegmentMax>();
    const auto data = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(ov::element::i64, ov::Shape{5});

    const auto op = std::make_shared<SegmentMax>(data, segment_ids, 0);
    NodeBuilder builder(op, {data, segment_ids});
    auto g_op = ov::as_type_ptr<SegmentMax>(builder.create());

    EXPECT_EQ(g_op->get_empty_segment_value(), op->get_empty_segment_value());
}

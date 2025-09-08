// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/segment_max.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {
using ov::op::v0::Parameter, ov::test::NodeBuilder;

TEST(attributes, segment_max_v16_with_num_segments) {
    NodeBuilder::opset().insert<ov::op::v16::SegmentMax>();
    const auto data = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(ov::element::i64, ov::Shape{3});
    const auto num_segments = std::make_shared<Parameter>(ov::element::i64, ov::Shape{});

    const auto op = std::make_shared<ov::op::v16::SegmentMax>(data, segment_ids, num_segments, ov::op::FillMode::ZERO);
    NodeBuilder builder(op, {data, segment_ids, num_segments});
    auto g_op = ov::as_type_ptr<ov::op::v16::SegmentMax>(builder.create());

    EXPECT_EQ(g_op->get_fill_mode(), op->get_fill_mode());
}

TEST(attributes, segment_max_v16_without_num_segments) {
    NodeBuilder::opset().insert<ov::op::v16::SegmentMax>();
    const auto data = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(ov::element::i64, ov::Shape{3});

    const auto op = std::make_shared<ov::op::v16::SegmentMax>(data, segment_ids, ov::op::FillMode::LOWEST);
    NodeBuilder builder(op, {data, segment_ids});
    auto g_op = ov::as_type_ptr<ov::op::v16::SegmentMax>(builder.create());

    EXPECT_EQ(g_op->get_fill_mode(), op->get_fill_mode());
}
}  // namespace ov::test

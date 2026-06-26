// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bincount.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, bincount_v17_op_unweighted_defaults) {
    NodeBuilder::opset().insert<ov::op::v17::Bincount>();
    auto data = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{10});
    auto bc = std::make_shared<ov::op::v17::Bincount>(data);
    NodeBuilder builder(bc, {data});

    auto g_bc = ov::as_type_ptr<ov::op::v17::Bincount>(builder.create());
    EXPECT_EQ(g_bc->get_minlength(), bc->get_minlength());
}

TEST(attributes, bincount_v17_op_with_minlength) {
    NodeBuilder::opset().insert<ov::op::v17::Bincount>();
    auto data = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{10});
    auto bc = std::make_shared<ov::op::v17::Bincount>(data, int64_t{5});
    NodeBuilder builder(bc, {data});

    auto g_bc = ov::as_type_ptr<ov::op::v17::Bincount>(builder.create());
    EXPECT_EQ(g_bc->get_minlength(), 5);
}

TEST(attributes, bincount_v17_op_weighted) {
    NodeBuilder::opset().insert<ov::op::v17::Bincount>();
    auto data = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{10});
    auto weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{10});
    auto bc = std::make_shared<ov::op::v17::Bincount>(data, weights, int64_t{3});
    NodeBuilder builder(bc, {data, weights});

    auto g_bc = ov::as_type_ptr<ov::op::v17::Bincount>(builder.create());
    EXPECT_EQ(g_bc->get_minlength(), bc->get_minlength());
}

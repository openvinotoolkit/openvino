// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_elements.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, gather_elements_op) {
    NodeBuilder::opset().insert<ov::op::v6::GatherElements>();
    auto arg1 = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{3});
    auto arg2 = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{7});
    int64_t axis = 0;

    auto gather_el = make_shared<ov::op::v6::GatherElements>(arg1, arg2, axis);
    NodeBuilder builder(gather_el, {arg1, arg2});
    auto g_gather_el = ov::as_type_ptr<ov::op::v6::GatherElements>(builder.create());

    EXPECT_EQ(g_gather_el->get_axis(), gather_el->get_axis());
}

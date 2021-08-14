// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/pad.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, pad_op) {
    NodeBuilder::get_ops().register_factory<op::v1::Pad>();
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});

    auto pad_mode = op::PadMode::EDGE;

    auto pad = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, pad_mode);
    NodeBuilder builder(pad);
    auto g_pad = as_type_ptr<op::v1::Pad>(builder.create());

    EXPECT_EQ(g_pad->get_pad_mode(), pad->get_pad_mode());
}

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, shuffle_channels_op)
{
    NodeBuilder::get_ops().register_factory<opset1::ShuffleChannels>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = 0;
    auto groups = 2;
    auto shuffle_channels = make_shared<opset1::ShuffleChannels>(data, axis, groups);
    NodeBuilder builder(shuffle_channels);
    auto g_shuffle_channels = as_type_ptr<opset1::ShuffleChannels>(builder.create());

    EXPECT_EQ(g_shuffle_channels->get_axis(), shuffle_channels->get_axis());
    EXPECT_EQ(g_shuffle_channels->get_group(), shuffle_channels->get_group());
}

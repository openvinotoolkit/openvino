// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, shuffle_channels_op)
{
    using ShuffleChannels = opset3::ShuffleChannels;

    NodeBuilder::get_ops().register_factory<ShuffleChannels>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 64, 16, 16});
    auto axis = 1;
    auto groups = 2;
    auto shuffle_channels = make_shared<ShuffleChannels>(data, axis, groups);
    NodeBuilder builder(shuffle_channels);
    auto g_shuffle_channels = as_type_ptr<ShuffleChannels>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_shuffle_channels->get_axis(), shuffle_channels->get_axis());
    EXPECT_EQ(g_shuffle_channels->get_group(), shuffle_channels->get_group());
}

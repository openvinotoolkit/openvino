// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/fake_quantize.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, fake_quantize_op) {
    NodeBuilder::get_ops().register_factory<op::v0::FakeQuantize>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});

    auto levels = 5;
    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    const auto fake_quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels, auto_broadcast);
    NodeBuilder builder(fake_quantize);
    auto g_fake_quantize = as_type_ptr<op::v0::FakeQuantize>(builder.create());

    // attribute count
    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_fake_quantize->get_levels(), fake_quantize->get_levels());
    EXPECT_EQ(g_fake_quantize->get_auto_broadcast(), fake_quantize->get_auto_broadcast());
}

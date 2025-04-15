// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_quantize.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, fake_quantize_op) {
    NodeBuilder::opset().insert<ov::op::v0::FakeQuantize>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto input_low = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});

    auto levels = 5;
    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    const auto fake_quantize =
        make_shared<op::v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels, auto_broadcast);
    NodeBuilder builder(fake_quantize, {data, input_low, input_high, output_low, output_high});
    auto g_fake_quantize = ov::as_type_ptr<op::v0::FakeQuantize>(builder.create());

    // attribute count
    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_fake_quantize->get_levels(), fake_quantize->get_levels());
    EXPECT_EQ(g_fake_quantize->get_auto_broadcast(), fake_quantize->get_auto_broadcast());
}

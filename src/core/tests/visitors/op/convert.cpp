// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, convert_op_v0) {
    using Convert = op::v0::Convert;

    NodeBuilder::opset().insert<Convert>();
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 4});
    const element::Type destination_type = element::Type_t::i32;

    const auto convert = std::make_shared<Convert>(data, destination_type);
    NodeBuilder builder(convert, {data});

    // attribute count
    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    // destination_type attribute
    const auto g_convert = ov::as_type_ptr<Convert>(builder.create());
    EXPECT_EQ(g_convert->get_destination_type(), convert->get_destination_type());
}

// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "util/visitor.hpp"

using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, scatter_nd_update) {
    NodeBuilder::get_ops().register_factory<opset4::ScatterNDUpdate>();

    auto data = std::make_shared<op::Parameter>(element::f32, Shape{1000, 256, 10, 15});
    auto indices = std::make_shared<op::Parameter>(element::i32, Shape{25, 125, 3});
    auto updates = std::make_shared<op::Parameter>(element::f32, Shape{25, 125, 15});

    auto scatter = std::make_shared<opset4::ScatterNDUpdate>(data, indices, updates);
    NodeBuilder builder(scatter, {data, indices, updates});
    EXPECT_NO_THROW(auto g_scatter = ov::as_type_ptr<opset4::ScatterNDUpdate>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

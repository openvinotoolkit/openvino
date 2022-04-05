// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "util/visitor.hpp"

using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, scatter_elements_update) {
    NodeBuilder::get_ops().register_factory<opset3::ScatterElementsUpdate>();

    auto data = std::make_shared<op::Parameter>(element::f32, Shape{2, 4, 5, 7});
    auto indices = std::make_shared<op::Parameter>(element::i16, Shape{2, 2, 2, 2});
    auto updates = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 2, 2});
    auto axis = std::make_shared<op::Parameter>(element::i16, Shape{});

    auto scatter = std::make_shared<opset3::ScatterElementsUpdate>(data, indices, updates, axis);
    NodeBuilder builder(scatter);

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

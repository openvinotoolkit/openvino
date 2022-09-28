// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ov;
using ngraph::test::NodeBuilder;

TEST(attributes, grid_sample_defaults) {
    NodeBuilder::get_ops().register_factory<opset9::GridSample>();
    const auto data = make_shared<opset9::Parameter>(element::f32, Shape{1, 3, 10, 10});
    const auto grid = make_shared<opset9::Parameter>(element::f32, Shape{1, 5, 5, 2});

    const auto op = make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});
    NodeBuilder builder(op);

    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

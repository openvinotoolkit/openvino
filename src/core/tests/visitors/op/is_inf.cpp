// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/opsets/opset10.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ov;
using ngraph::test::NodeBuilder;

TEST(attributes, is_inf_defaults) {
    NodeBuilder::get_ops().register_factory<opset10::IsInf>();
    const auto data = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 10, 10});

    const auto op = make_shared<opset10::IsInf>(data, opset10::IsInf::Attributes{});
    NodeBuilder builder(op);

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size()y, expected_attr_count);
}

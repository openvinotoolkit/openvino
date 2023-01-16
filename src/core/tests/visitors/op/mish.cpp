// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, mish_op) {
    NodeBuilder::get_ops().register_factory<opset4::Mish>();
    const auto A = make_shared<op::Parameter>(element::f32, Shape{5, 2});

    const auto mish = make_shared<opset4::Mish>(A);
    NodeBuilder builder(mish, {A});
    EXPECT_NO_THROW(auto g_mish = ov::as_type_ptr<opset4::Mish>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

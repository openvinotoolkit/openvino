// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, erf_op)
{
    NodeBuilder::get_ops().register_factory<opset7::Erf>();
    const auto A = make_shared<op::Parameter>(element::f32, Shape{3, 7});

    const auto erf = make_shared<opset7::Erf>(A);
    NodeBuilder builder(erf);

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

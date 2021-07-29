// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, result)
{
    const auto arg_shape =Shape{1, 2, 3, 4, 5};
    auto arg = make_shared<opset1::Constant>(element::f32, arg_shape);

    auto result = make_shared<opset1::Result>(arg);

    EXPECT_EQ(result->get_output_element_type(0), element::f32);
    EXPECT_EQ(result->get_output_shape(0), arg_shape);
}

TEST(type_prop, result_dynamic_shape)
{
    auto arg = make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());

    auto result = make_shared<opset1::Result>(arg);

    EXPECT_EQ(result->get_output_element_type(0), element::f32);
    EXPECT_TRUE(result->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

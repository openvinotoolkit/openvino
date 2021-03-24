// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, softmax_default_axis)
{
    const Shape arg_shape{2, 3};
    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto sm = make_shared<op::v1::Softmax>(arg);
    ASSERT_EQ(sm->get_axis(), 1);
}

TEST(type_prop, softmax_out_of_bound_axis)
{
    const Shape arg_shape{2, 3};
    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    // axis cannot be a negative number
    ASSERT_THROW(make_shared<op::v1::Softmax>(arg, -1), ngraph::NodeValidationFailure);
}

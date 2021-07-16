// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, random_uniform_test)
{
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = opset8::Constant::create(element::f32, Shape{}, 0);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1);

    auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 3, 4, 5}));
}

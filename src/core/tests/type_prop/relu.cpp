// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, relu_2d) {
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    Shape relu_shape{2, 4};
    auto relu = make_shared<op::Relu>(param);
    ASSERT_EQ(relu->get_element_type(), element::f32);
    ASSERT_EQ(relu->get_shape(), relu_shape);
}

TEST(type_prop, relu_4d) {
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2, 2});
    Shape relu_shape{2, 2, 2, 2};
    auto relu = make_shared<op::Relu>(param);
    ASSERT_EQ(relu->get_element_type(), element::f32);
    ASSERT_EQ(relu->get_shape(), relu_shape);
}
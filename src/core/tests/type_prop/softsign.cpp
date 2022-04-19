// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, softsign) {
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto softsign_func = make_shared<op::v9::SoftSign>(data);
    EXPECT_EQ(softsign_func->get_element_type(), element::f32);
    EXPECT_EQ(softsign_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, softsign_partial) {
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto softsign_func = make_shared<op::v9::SoftSign>(data);
    EXPECT_EQ(softsign_func->get_element_type(), element::f32);
    ASSERT_TRUE(softsign_func->get_output_partial_shape(0).same_scheme((PartialShape{1, Dimension::dynamic(), 6})));
    ASSERT_TRUE(softsign_func->get_output_partial_shape(0).rank().is_static());

    // rank unknown
    auto softsign_partial =
        make_shared<op::v9::SoftSign>(make_shared<op::Parameter>(element::f32, PartialShape::dynamic()));
    ASSERT_TRUE(softsign_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

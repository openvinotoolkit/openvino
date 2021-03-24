// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, unsqueeze)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{1, 2});
    auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);

    ASSERT_EQ(unsqueeze->get_element_type(), element::f32);
    ASSERT_EQ(unsqueeze->get_shape(), (Shape{4, 1, 1, 1, 4, 1, 8}));
}

TEST(type_prop, unsqueeze_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(5));
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{1, 2});
    auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);

    ASSERT_EQ(unsqueeze->get_element_type(), element::f32);
    EXPECT_TRUE(
        unsqueeze->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(),
                                                                        1,
                                                                        1,
                                                                        Dimension::dynamic(),
                                                                        Dimension::dynamic(),
                                                                        Dimension::dynamic(),
                                                                        Dimension::dynamic()}));
}

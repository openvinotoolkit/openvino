//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace ngraph;
using namespace std;

TEST(node_input_output, input_create)
{
    auto x = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto add = make_shared<op::Add>(x, y);

    auto add_in_0 = add->input(0);
    auto add_in_1 = add->input(1);

    EXPECT_EQ(add_in_0.get_node(), add.get());
    EXPECT_EQ(add_in_0.get_index(), 0);
    EXPECT_EQ(add_in_0.get_element_type(), element::f32);
    EXPECT_EQ(add_in_0.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_in_0.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));
    EXPECT_EQ(add_in_0.get_source_output(), Output<Node>(x, 0));

    EXPECT_EQ(add_in_1.get_node(), add.get());
    EXPECT_EQ(add_in_1.get_index(), 1);
    EXPECT_EQ(add_in_1.get_element_type(), element::f32);
    EXPECT_EQ(add_in_1.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_in_1.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));
    EXPECT_EQ(add_in_1.get_source_output(), Output<Node>(y, 0));

    EXPECT_THROW(add->input(2), std::out_of_range);
}

TEST(node_input_output, input_create_const)
{
    auto x = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto add = make_shared<const op::Add>(x, y);

    auto add_in_0 = add->input(0);
    auto add_in_1 = add->input(1);

    EXPECT_EQ(add_in_0.get_node(), add.get());
    EXPECT_EQ(add_in_0.get_index(), 0);
    EXPECT_EQ(add_in_0.get_element_type(), element::f32);
    EXPECT_EQ(add_in_0.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_in_0.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));
    EXPECT_EQ(add_in_0.get_source_output(), Output<Node>(x, 0));

    EXPECT_EQ(add_in_1.get_node(), add.get());
    EXPECT_EQ(add_in_1.get_index(), 1);
    EXPECT_EQ(add_in_1.get_element_type(), element::f32);
    EXPECT_EQ(add_in_1.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_in_1.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));
    EXPECT_EQ(add_in_1.get_source_output(), Output<Node>(y, 0));

    EXPECT_THROW(add->input(2), std::out_of_range);
}

TEST(node_input_output, output_create)
{
    auto x = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto add = make_shared<op::Add>(x, y);

    auto add_out_0 = add->output(0);

    EXPECT_EQ(add_out_0.get_node(), add.get());
    EXPECT_EQ(add_out_0.get_index(), 0);
    EXPECT_EQ(add_out_0.get_element_type(), element::f32);
    EXPECT_EQ(add_out_0.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_out_0.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));

    EXPECT_THROW(add->output(1), std::out_of_range);
}

TEST(node_input_output, output_create_const)
{
    auto x = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto add = make_shared<const op::Add>(x, y);

    auto add_out_0 = add->output(0);

    EXPECT_EQ(add_out_0.get_node(), add.get());
    EXPECT_EQ(add_out_0.get_index(), 0);
    EXPECT_EQ(add_out_0.get_element_type(), element::f32);
    EXPECT_EQ(add_out_0.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_out_0.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));

    EXPECT_THROW(add->output(1), std::out_of_range);
}

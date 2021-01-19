//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

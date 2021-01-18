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

TEST(type_prop, param_partial_rank_dynamic)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto& pshape = a->get_output_partial_shape(0);

    ASSERT_TRUE(pshape.is_dynamic());
    ASSERT_TRUE(pshape.rank().is_dynamic());
}

TEST(type_prop, param_partial_rank_static)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3, 4});

    auto& pshape = a->get_output_partial_shape(0);

    ASSERT_TRUE(pshape.is_dynamic());
    ASSERT_EQ(pshape.rank().get_length(), 4);
    ASSERT_TRUE(pshape[0].is_static() && pshape[0].get_length() == 2);
    ASSERT_TRUE(pshape[1].is_dynamic());
    ASSERT_TRUE(pshape[2].is_static() && pshape[2].get_length() == 3);
    ASSERT_TRUE(pshape[3].is_static() && pshape[3].get_length() == 4);
}

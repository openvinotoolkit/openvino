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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, prelu)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto slope = make_shared<op::Parameter>(element::f32, Shape{2});
    Shape prelu_shape{2, 4};
    auto prelu = make_shared<op::PRelu>(param, slope);
    ASSERT_EQ(prelu->get_element_type(), element::f32);
    ASSERT_EQ(prelu->get_shape(), prelu_shape);
}

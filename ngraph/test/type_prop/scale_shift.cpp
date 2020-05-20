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

TEST(type_prop, scale_shift_no_broadcast)
{
    auto data = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto scale = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto shift = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto scale_shift_func = make_shared<op::ScaleShift>(data, scale, shift);
    EXPECT_EQ(scale_shift_func->get_element_type(), element::f64);
    EXPECT_EQ(scale_shift_func->get_shape(), (Shape{3, 6}));
}

TEST(type_prop, scale_shift)
{
    auto data = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto scale = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto shift = make_shared<op::Parameter>(element::f64, Shape{});
    auto scale_shift_func = make_shared<op::ScaleShift>(data, scale, shift);
    EXPECT_EQ(scale_shift_func->get_element_type(), element::f64);
    EXPECT_EQ(scale_shift_func->get_shape(), (Shape{3, 6}));
}

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

using namespace std;
using namespace ngraph;

TEST(type_prop, softmax_default_axis)
{
    const Shape arg_shape{2, 3};
    auto arg = make_shared<op::Parameter>(element::Type_t::f32, arg_shape);
    auto sm = make_shared<op::v1::Softmax>(arg);
    ASSERT_EQ(sm->get_axis(), 1);
}

TEST(type_prop, softmax_out_of_bound_axis)
{
    const Shape arg_shape{2, 3};
    auto arg = make_shared<op::Parameter>(element::Type_t::f32, arg_shape);
    // axis cannot be a negative number
    ASSERT_THROW(make_shared<op::v1::Softmax>(arg, -1), ngraph::NodeValidationFailure);
}

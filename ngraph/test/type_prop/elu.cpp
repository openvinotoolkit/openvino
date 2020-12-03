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

TEST(type_prop, elu)
{
    Shape data_shape{2, 4};
    auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);
    auto elu = make_shared<op::Elu>(data, 1);
    ASSERT_EQ(elu->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(elu->get_shape(), data_shape);
}

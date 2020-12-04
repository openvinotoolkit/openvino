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

TEST(type_prop, hardsigmoid)
{
    const Shape data_shape{3, 5};

    const auto P = make_shared<op::Parameter>(element::f32, data_shape);
    const auto alpha = op::Constant::create<float>(P->get_element_type(), Shape{}, {0.1f});
    const auto beta = op::Constant::create<float>(P->get_element_type(), Shape{}, {1.2f});
    const auto H = make_shared<op::HardSigmoid>(P, alpha, beta);
    ASSERT_EQ(H->get_element_type(), element::f32);
    ASSERT_EQ(H->get_shape(), data_shape);
}

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

TEST(type_prop, fused_clamp)
{
    const auto data = make_shared<op::Parameter>(element::f64, Shape{2, 2});

    try
    {
        const auto clamp = make_shared<op::Clamp>(data, 2.0, 1.0);
        EXPECT_FALSE(clamp.get())
            << "Clamp validation did not work. Op node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("The 'min' parameter needs to be less than 'max' for Clamp"));
    }

    const auto clamp = make_shared<op::Clamp>(data, 1.0, 2.0);
    EXPECT_EQ(clamp->get_element_type(), element::f64);
    EXPECT_EQ(clamp->get_shape(), (Shape{2, 2}));
}

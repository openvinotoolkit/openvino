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

TEST(type_prop, squared_difference)
{
    const auto x1 = make_shared<op::Parameter>(element::f64, Shape{2, 2});
    const auto x2 = make_shared<op::Parameter>(element::f64, Shape{3, 2});
    const auto x3 = make_shared<op::Parameter>(element::f64, Shape{1, 2});

    try
    {
        const auto squared_diff = make_shared<op::SquaredDifference>(x1, x2);
        FAIL() << "SquaredDifference node was created with incorrect data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }

    const auto clamp = make_shared<op::SquaredDifference>(x1, x3);
    EXPECT_EQ(clamp->get_element_type(), element::f64);
    EXPECT_EQ(clamp->get_shape(), (Shape{2, 2}));
}

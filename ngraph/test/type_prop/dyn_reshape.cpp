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

TEST(type_prop, reshape_v1_arg_rank_static_pattern_zero)
{
    auto arg = make_shared<op::Parameter>(element::Type_t::f32, Shape{2, 0, 2, 8});
    auto pattern = op::Constant::create(element::Type_t::i64, Shape{4}, {1, 2, 0, 32});

    auto reshape_v1_static = make_shared<op::v1::Reshape>(arg, pattern, true);
    EXPECT_EQ(reshape_v1_static->get_output_shape(0), Shape({1, 2, 2, 32}));

    auto dynamic_arg = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto reshape_v1_dynamic = make_shared<op::v1::Reshape>(dynamic_arg, pattern, true);
    EXPECT_TRUE(reshape_v1_dynamic->get_output_partial_shape(0).same_scheme(
        PartialShape{1, 2, Dimension::dynamic(), 32}));
    try
    {
        auto static_shape_parameter =
            make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 2, 3, 4});
        auto reshape_output_pattern =
            op::Constant::create(element::Type_t::i64, Shape{4}, {2, 2, 3, 4});
        auto reshape =
            make_shared<op::v1::Reshape>(static_shape_parameter, reshape_output_pattern, true);
        FAIL() << "Expected failure on reshape construction";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("is incompatible with input shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

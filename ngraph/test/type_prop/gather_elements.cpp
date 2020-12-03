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

// ------------------------------ V6 ------------------------------

TEST(type_prop, gather_elements_2d_axis_0)
{
    Shape data_shape{3, 3};
    Shape indices_shape{2, 3};
    int axis = 0;
    auto D = make_shared<op::Parameter>(element::Type_t::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::Type_t::i32, indices_shape);
    auto GE = make_shared<op::v6::GatherElements>(D, I, axis);
    ASSERT_EQ(GE->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(GE->get_shape(), indices_shape);
}

TEST(type_prop, gather_elements_2d_axis_1)
{
    Shape data_shape{3, 3};
    Shape indices_shape{3, 1};
    int axis = 1;
    auto D = make_shared<op::Parameter>(element::Type_t::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::Type_t::i32, indices_shape);
    auto GE = make_shared<op::v6::GatherElements>(D, I, axis);
    ASSERT_EQ(GE->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(GE->get_shape(), indices_shape);
}

TEST(type_prop, gather_elements_3d_axis_0)
{
    Shape data_shape{3, 3, 10000};
    Shape indices_shape{300, 3, 10000};
    int axis = 0;
    auto D = make_shared<op::Parameter>(element::Type_t::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::Type_t::i32, indices_shape);
    auto GE = make_shared<op::v6::GatherElements>(D, I, axis);
    ASSERT_EQ(GE->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(GE->get_shape(), indices_shape);
}

// --------------------- Negative tests ------------------------------
TEST(type_prop, gather_elements_shapes_inconsistency)
{
    Shape data_shape{3, 3};
    Shape indices_shape{2, 1};
    int axis = 1;
    auto D = make_shared<op::Parameter>(element::Type_t::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::Type_t::i32, indices_shape);

    try
    {
        auto GE = make_shared<op::v6::GatherElements>(D, I, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Shape inconsistency check failed";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("data and indices must have equal shapes except for axis "));
    }
    catch (...)
    {
        FAIL() << "Deduced shape check failed for unexpected reason";
    }
}

TEST(type_prop, gather_elements_type_inconsistency)
{
    Shape data_shape{3, 3};
    Shape indices_shape{2, 1};
    int axis = 1;
    auto D = make_shared<op::Parameter>(element::Type_t::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::Type_t::u32, indices_shape);

    try
    {
        auto GE = make_shared<op::v6::GatherElements>(D, I, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "the indices tensor type check failed";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("indices mush be of int32 or int64 type. But instead got"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

// negative tests
// axis out of bounds
// rank check
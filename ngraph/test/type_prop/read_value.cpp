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

TEST(type_prop, read_value_deduce_basic)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto read_value = make_shared<op::v3::ReadValue>(input, "variable_id");

    ASSERT_EQ(read_value->get_element_type(), element::f32);
    ASSERT_EQ(read_value->get_shape(), (Shape{1, 2, 64, 64}));
    ASSERT_EQ(read_value->get_variable_id(), "variable_id");

    auto var_info = read_value->get_variable();
    ASSERT_EQ(var_info->get_info().data_type, element::f32);
    ASSERT_EQ(var_info->get_info().data_shape, (Shape{1, 2, 64, 64}));
    ASSERT_EQ(var_info->get_info().variable_id, "variable_id");
}

TEST(type_prop, read_value_deduce_dynamic)
{
    auto input =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic(), 2, 64, 64});
    auto read_value = make_shared<op::v3::ReadValue>(input, "variable_id");

    ASSERT_EQ(read_value->get_element_type(), element::i64);
    ASSERT_TRUE(read_value->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, 64, 64}));
    ASSERT_EQ(read_value->get_variable_id(), "variable_id");

    auto var_info = read_value->get_variable();
    ASSERT_EQ(var_info->get_info().data_type, element::i64);
    ASSERT_TRUE(
        var_info->get_info().data_shape.same_scheme(PartialShape{Dimension::dynamic(), 2, 64, 64}));
    ASSERT_EQ(var_info->get_info().variable_id, "variable_id");
}

TEST(type_prop, read_value_invalid_variable_id)
{
    auto input = make_shared<op::Parameter>(element::f16, Shape{1, 2, 64, 64});
    try
    {
        auto read_value = make_shared<op::v3::ReadValue>(input, "");
        // variable_id attribute cannot be an empty string
        FAIL() << "Exception should be thrown";
    }
    catch (const NodeValidationFailure error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Variable identifier may not be an empty string.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

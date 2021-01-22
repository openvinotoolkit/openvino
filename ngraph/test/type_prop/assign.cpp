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

TEST(type_prop, assign_invalid_variable_id)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    try
    {
        auto assign = make_shared<op::v3::Assign>(A, "");
        // Should have thrown, so fail if it didn't
        FAIL() << "Should not find variable with variable_id";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Variable identifier attribute may not be an empty string."));
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, assign_variable_not_found)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    try
    {
        auto assign = make_shared<op::v3::Assign>(A, "variable_id");
        // Should have thrown, so fail if it didn't
        FAIL() << "Should not find variable with variable_id";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Can't find variable with id = variable_id"));
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, assign_deduce_shape_basic)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto read_value = make_shared<op::v3::ReadValue>(input, "variable_id");
    auto assign = make_shared<op::v3::Assign>(read_value, "variable_id");

    ASSERT_EQ(assign->get_element_type(), element::f32);
    ASSERT_EQ(assign->get_shape(), (Shape{1, 2, 64, 64}));
    ASSERT_EQ(assign->get_variable_id(), "variable_id");
}

TEST(type_prop, assign_deduce_shape_dynamic)
{
    PartialShape input_pshape = PartialShape::dynamic();
    auto input = make_shared<op::Parameter>(element::f32, input_pshape);
    auto read_value = make_shared<op::v3::ReadValue>(input, "variable_id");

    // Update PartialShape of ReadValue variable to see if Assign op produces
    // output PartialShape with most possibly specialized value (as least number of dynamic
    // dimensions as possible)
    PartialShape var_pshape{Dimension::dynamic(), 2, 64, 64};
    VariableInfo info = {var_pshape, element::f32, "variable_id"};
    auto var_info = make_shared<Variable>(info);
    read_value->set_variable(var_info);

    auto assign = make_shared<op::v3::Assign>(read_value, "variable_id");

    ASSERT_EQ(assign->get_element_type(), element::f32);
    ASSERT_TRUE(assign->get_output_partial_shape(0).same_scheme(var_pshape));
}

TEST(type_prop, assign_incompatible_variables)
{
    auto input =
        make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 64, 64});
    auto read_value = make_shared<op::v3::ReadValue>(input, "variable_id");
    VariableInfo info = {PartialShape{2, 2, 64, 64}, element::f32, "variable_id"};
    auto var_info = make_shared<Variable>(info);

    // Incompatible partial shape
    read_value->set_variable(var_info);
    try
    {
        auto assign = make_shared<op::v3::Assign>(read_value, "variable_id");
        FAIL() << "Exception should be thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Variables output shapes are inconsistent.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }

    // Incompatible element type
    var_info->update(
        VariableInfo{PartialShape{Dimension::dynamic(), 2, 64, 64}, element::f16, "variable_id"});
    try
    {
        auto assign = make_shared<op::v3::Assign>(read_value, "variable_id");
        FAIL() << "Exception should be thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Variables types are inconsistent.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }

    // Incompatible variable identifier
    var_info->update(
        VariableInfo{PartialShape{Dimension::dynamic(), 2, 64, 64}, element::f32, "var_id"});
    try
    {
        auto assign = make_shared<op::v3::Assign>(read_value, "variable_id");
        FAIL() << "Exception should be thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Variables identifiers are inconsistent.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

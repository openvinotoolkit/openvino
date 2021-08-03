// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, assign_variable_not_found)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    try
    {
        auto space_to_depth = make_shared<opset5::Assign>(A, "variable_id");
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
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, assign_deduce)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto read_value = make_shared<opset5::ReadValue>(input, "variable_id");
    auto assign = make_shared<opset5::Assign>(read_value, "variable_id");

    ASSERT_EQ(assign->get_element_type(), element::f32);
    ASSERT_EQ(assign->get_shape(), (Shape{1, 2, 64, 64}));
}

TEST(type_prop, assign_read_value_new_shape)
{
    auto input = make_shared<op::Parameter>(element::f16, Shape{4, 3, 2, 1});

    auto variable =
        std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "ID"});
    auto read_value = make_shared<opset6::ReadValue>(input, variable);
    auto assign = make_shared<opset6::Assign>(read_value, variable);

    ASSERT_EQ(assign->get_element_type(), element::f16);
    ASSERT_EQ(assign->get_shape(), (Shape{4, 3, 2, 1}));

    auto f = std::make_shared<Function>(ResultVector{}, SinkVector{assign}, ParameterVector{input});

    input->set_partial_shape({3, {4, 5}, 8});
    f->validate_nodes_and_infer_types();

    ASSERT_EQ(assign->get_element_type(), element::f16);
    ASSERT_EQ(assign->get_output_partial_shape(0), (PartialShape{3, {4, 5}, 8}));
    ASSERT_EQ(variable->get_info().data_type, element::f16);
    ASSERT_EQ(variable->get_info().data_shape, (PartialShape{3, {4, 5}, 8}));
}

TEST(type_prop, variable_comparison)
{
    auto variable1 =
        std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "ID"});

    auto variable2 =
        std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "ID"});

    auto variable3 =
        std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "ID1"});

    auto variable4 =
        std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::f32, "ID"});

    auto variable5 =
        std::make_shared<Variable>(VariableInfo{Shape{1}, element::dynamic, "ID"});

    ASSERT_TRUE(variable1->get_info() == variable2->get_info());
    ASSERT_FALSE(variable1->get_info() == variable3->get_info());
    ASSERT_FALSE(variable1->get_info() == variable4->get_info());
    ASSERT_FALSE(variable1->get_info() == variable5->get_info());
}
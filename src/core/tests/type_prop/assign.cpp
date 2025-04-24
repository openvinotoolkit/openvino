// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/assign.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/util/variable.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, assign_variable_not_found) {
    auto A = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 64, 64});
    try {
        auto space_to_depth = make_shared<ov::op::v3::Assign>(A, "variable_id");
        // Should have thrown, so fail if it didn't
        FAIL() << "Should not find variable with variable_id";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Can't find variable with id = variable_id"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, assign_deduce) {
    auto input = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 64, 64});
    auto read_value = make_shared<ov::op::v3::ReadValue>(input, "variable_id");
    auto assign = make_shared<ov::op::v3::Assign>(read_value, "variable_id");

    ASSERT_EQ(assign->get_element_type(), ov::element::f32);
    ASSERT_EQ(assign->get_shape(), (ov::Shape{1, 2, 64, 64}));
}

TEST(type_prop, assign_set_new_shape_allowed_range) {
    auto input = make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{4, 3, 2, 1});

    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "ID"});
    auto read_value = make_shared<ov::op::v6::ReadValue>(input, variable);
    auto assign = make_shared<ov::op::v6::Assign>(read_value, variable);

    ASSERT_EQ(assign->get_element_type(), ov::element::f16);
    ASSERT_EQ(assign->get_output_partial_shape(0), (ov::PartialShape{4, 3, 2, 1}));

    auto m = std::make_shared<ov::Model>(ov::ResultVector{}, ov::SinkVector{assign}, ov::ParameterVector{input});

    input->set_partial_shape({3, {4, 5}, 8});
    m->validate_nodes_and_infer_types();

    ASSERT_EQ(assign->get_element_type(), ov::element::f16);
    ASSERT_EQ(assign->get_output_partial_shape(0), (ov::PartialShape{3, {4, 5}, 8}));
    ASSERT_EQ(variable->get_info().data_type, ov::element::dynamic);
    ASSERT_EQ(variable->get_info().data_shape, (ov::PartialShape::dynamic()));
}

TEST(type_prop, variable_comparison) {
    auto variable1 = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "ID"});

    auto variable2 = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "ID"});

    auto variable3 = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "ID1"});

    auto variable4 = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::f32, "ID"});

    auto variable5 =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{ov::Shape{1}, ov::element::dynamic, "ID"});

    ASSERT_TRUE(variable1->get_info() == variable2->get_info());
    ASSERT_FALSE(variable1->get_info() == variable3->get_info());
    ASSERT_FALSE(variable1->get_info() == variable4->get_info());
    ASSERT_FALSE(variable1->get_info() == variable5->get_info());
}

TEST(type_prop, assign_v6_static_shape_match) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto variable = std::make_shared<op::util::Variable>(
        op::util::VariableInfo{PartialShape{1, 2, 64, 64}, element::f32, "variable_id"});
    std::shared_ptr<ov::op::v6::Assign> assign;
    EXPECT_NO_THROW(assign = std::make_shared<ov::op::v6::Assign>(input, variable));

    ASSERT_EQ(assign->get_element_type(), element::f32);
    ASSERT_EQ(assign->get_shape(), (Shape{1, 2, 64, 64}));
}

TEST(type_prop, assign_v6_static_shapes_do_not_match) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 64, 64});
    auto variable_info = op::util::VariableInfo{PartialShape{1, 2, 64, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);
    std::shared_ptr<ov::op::v6::Assign> assign;
    EXPECT_ANY_THROW(assign = std::make_shared<ov::op::v6::Assign>(input, variable));
}

TEST(type_prop, assign_v6_static_types_do_not_match) {
    auto input = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 2, 64, 64});
    auto variable_info = op::util::VariableInfo{PartialShape{1, 2, 64, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);
    std::shared_ptr<ov::op::v6::Assign> assign;
    EXPECT_ANY_THROW(assign = std::make_shared<ov::op::v6::Assign>(input, variable));
}

TEST(type_prop, assign_v6_dyn_shape_type_in_variable) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 64, 64});

    auto variable_info = op::util::VariableInfo{PartialShape{Dimension::dynamic(), 2, Dimension::dynamic(), 64},
                                                element::dynamic,
                                                "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);

    std::shared_ptr<ov::op::v6::Assign> assign;
    EXPECT_NO_THROW(assign = std::make_shared<ov::op::v6::Assign>(input, variable));

    ASSERT_EQ(assign->get_element_type(), element::f32);
    ASSERT_EQ(assign->get_output_partial_shape(0), (PartialShape{1, 2, 64, 64}));
    ASSERT_EQ(assign->get_variable_id(), "variable_id");
}

TEST(type_prop, assign_v6_dyn_type_in_input) {
    auto input = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{1, 2, 64, 64});

    auto variable_info = op::util::VariableInfo{PartialShape{1, 2, 64, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);

    std::shared_ptr<ov::op::v6::Assign> assign;
    EXPECT_NO_THROW(assign = std::make_shared<ov::op::v6::Assign>(input, variable));

    ASSERT_EQ(assign->get_element_type(), element::dynamic);
    ASSERT_EQ(assign->get_output_partial_shape(0), (PartialShape{1, 2, 64, 64}));
    ASSERT_EQ(assign->get_variable_id(), "variable_id");
}

TEST(type_prop, assign_v6_init_shape_is_in_range) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 2, 64, 64});

    auto variable_info =
        op::util::VariableInfo{PartialShape{{1, 10}, {2, 5}, {64, 64}, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);

    std::shared_ptr<ov::op::v6::Assign> assign;
    EXPECT_NO_THROW(assign = std::make_shared<ov::op::v6::Assign>(input, variable));

    ASSERT_EQ(assign->get_element_type(), element::f32);
    ASSERT_EQ(assign->get_output_partial_shape(0), (PartialShape{1, 2, 64, 64}));
    ASSERT_EQ(assign->get_variable_id(), "variable_id");
}

TEST(type_prop, assign_v6_init_shape_is_not_in_range) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 2, 64, 64});

    auto variable_info = op::util::VariableInfo{PartialShape{{2, 5}, {2, 5}, 64, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);

    std::shared_ptr<ov::op::v6::Assign> assign;
    EXPECT_ANY_THROW(assign = std::make_shared<ov::op::v6::Assign>(input, variable));
}

TEST(type_prop, assign_v6_init_shape_is_in_range_2) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{{1, 2}, 2, 64, 64});

    auto variable_info = op::util::VariableInfo{PartialShape{1, 2, 64, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);

    std::shared_ptr<ov::op::v6::Assign> assign;
    EXPECT_NO_THROW(assign = std::make_shared<ov::op::v6::Assign>(input, variable));

    ASSERT_EQ(assign->get_element_type(), element::f32);
    ASSERT_EQ(assign->get_output_partial_shape(0), (PartialShape{{1, 2}, 2, 64, 64}));
    ASSERT_EQ(assign->get_variable_id(), "variable_id");
}

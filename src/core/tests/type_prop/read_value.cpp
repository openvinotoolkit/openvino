// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/read_value.hpp"

#include "common_test_utils/type_prop.hpp"
#include "dimension_util.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, read_value_deduce) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto read_value = make_shared<ov::op::v3::ReadValue>(input, "variable_id");

    ASSERT_EQ(read_value->get_element_type(), element::f32);
    ASSERT_EQ(read_value->get_shape(), (Shape{1, 2, 64, 64}));
    ASSERT_EQ(read_value->get_variable_id(), "variable_id");
}

TEST(type_prop, read_value_v6_static_shape_match) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto variable = std::make_shared<op::util::Variable>(
        op::util::VariableInfo{PartialShape{1, 2, 64, 64}, element::f32, "variable_id"});
    std::shared_ptr<ov::op::v6::ReadValue> read_value;
    EXPECT_NO_THROW(read_value = std::make_shared<ov::op::v6::ReadValue>(input, variable));

    ASSERT_EQ(read_value->get_element_type(), element::f32);
    ASSERT_EQ(read_value->get_shape(), (Shape{1, 2, 64, 64}));
}

TEST(type_prop, read_value_v6_static_shapes_do_not_match) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 64, 64});
    auto variable_info = op::util::VariableInfo{PartialShape{1, 2, 64, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);
    std::shared_ptr<ov::op::v6::ReadValue> read_value;
    EXPECT_ANY_THROW(read_value = std::make_shared<ov::op::v6::ReadValue>(input, variable));
}

TEST(type_prop, read_value_v6_static_types_do_not_match) {
    auto input = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 2, 64, 64});
    auto variable_info = op::util::VariableInfo{PartialShape{1, 2, 64, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);
    std::shared_ptr<ov::op::v6::ReadValue> read_value;
    EXPECT_ANY_THROW(read_value = std::make_shared<ov::op::v6::ReadValue>(input, variable));
}

TEST(type_prop, read_value_v6_dyn_shape_type_in_variable) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 64, 64});

    auto variable_info = op::util::VariableInfo{PartialShape{Dimension::dynamic(), 2, Dimension::dynamic(), 64},
                                                element::dynamic,
                                                "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);

    std::shared_ptr<ov::op::v6::ReadValue> read_value;
    EXPECT_NO_THROW(read_value = std::make_shared<ov::op::v6::ReadValue>(input, variable));

    ASSERT_EQ(read_value->get_element_type(), element::dynamic);
    ASSERT_EQ(read_value->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), 2, Dimension::dynamic(), 64}));
    ASSERT_EQ(read_value->get_variable_id(), "variable_id");
}

TEST(type_prop, read_value_v6_init_shape_is_in_range) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 2, 64, 64});

    auto variable_info =
        op::util::VariableInfo{PartialShape{{1, 10}, {2, 5}, {64, 64}, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);

    std::shared_ptr<ov::op::v6::ReadValue> read_value;
    EXPECT_NO_THROW(read_value = std::make_shared<ov::op::v6::ReadValue>(input, variable));

    ASSERT_EQ(read_value->get_element_type(), element::f32);
    ASSERT_EQ(read_value->get_output_partial_shape(0), (PartialShape{{1, 10}, {2, 5}, {64, 64}, 64}));
    ASSERT_EQ(read_value->get_variable_id(), "variable_id");
}

TEST(type_prop, read_value_v6_init_shape_is_not_in_range) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 2, 64, 64});

    auto variable_info = op::util::VariableInfo{PartialShape{{2, 5}, {2, 5}, 64, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);

    std::shared_ptr<ov::op::v6::ReadValue> read_value;
    EXPECT_ANY_THROW(read_value = std::make_shared<ov::op::v6::ReadValue>(input, variable));
}

TEST(type_prop, read_value_v6_init_shape_is_not_in_range_2) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{{1, 2}, 2, 64, 64});

    auto variable_info = op::util::VariableInfo{PartialShape{1, 2, 64, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);

    std::shared_ptr<ov::op::v6::ReadValue> read_value;
    EXPECT_ANY_THROW(read_value = std::make_shared<ov::op::v6::ReadValue>(input, variable));
}

TEST(type_prop, read_value_v6_no_init) {
    auto variable_info = op::util::VariableInfo{PartialShape{1, 2, 64, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);

    std::shared_ptr<ov::op::v6::ReadValue> read_value = std::make_shared<ov::op::v6::ReadValue>(variable);
    ASSERT_EQ(read_value->get_element_type(), element::f32);
    ASSERT_EQ(read_value->get_output_partial_shape(0), (PartialShape{1, 2, 64, 64}));
    ASSERT_EQ(read_value->get_variable_id(), "variable_id");
}

TEST(type_prop, read_value_symbols_propagation) {
    auto variable_pshape = PartialShape{1, 2, 64, 64};
    auto symbols = set_shape_symbols(variable_pshape);
    auto variable_info = op::util::VariableInfo{variable_pshape, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);
    std::shared_ptr<ov::op::v6::ReadValue> read_value = std::make_shared<ov::op::v6::ReadValue>(variable);
    EXPECT_THAT(get_shape_symbols(read_value->get_output_partial_shape(0)), symbols);
}

TEST(type_prop, DISABLED_read_value_symbols_propagation_from_init_subgraph) {
    auto input_pshape = PartialShape{1, 2, 64, 64};
    auto symbols = set_shape_symbols(input_pshape);
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, input_pshape);
    auto variable_info = op::util::VariableInfo{{1, 2, 64, 64}, element::f32, "variable_id"};
    auto variable = std::make_shared<op::util::Variable>(variable_info);
    std::shared_ptr<ov::op::v6::ReadValue> read_value = std::make_shared<ov::op::v6::ReadValue>(input, variable);
    EXPECT_THAT(get_shape_symbols(read_value->get_output_partial_shape(0)), symbols);
}

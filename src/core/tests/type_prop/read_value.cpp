// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/read_value.hpp"

#include "common_test_utils/type_prop.hpp"
#include "dimension_util.hpp"

#include "openvino/op/read_value.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/core.hpp"

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


TEST(type_prop, my_test) {
    using namespace ov::pass::pattern;
    using namespace ov;

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1});
    position_ids->output(0).set_names({"position_ids"});

    auto var_info = ov::op::util::VariableInfo{ov::PartialShape{-1}, ov::element::i64, "var_" + std::to_string(0)};
    auto var = std::make_shared<ov::op::util::Variable>(var_info);
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(ov::op::v0::Constant::create(ov::element::i64, ov::Shape{0}, {}), var);
    auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{read_value->output(0), position_ids->output(0)}, 0);
    auto assign = std::make_shared<ov::op::v6::Assign>(concat, var);
    
    auto res = std::make_shared<ov::op::v0::Result>(concat);

    auto model = std::make_shared<ov::Model>(ResultVector{res}, SinkVector{assign}, ParameterVector{position_ids});

    ov::Core core;
    auto compiled_model = core.compile_model(model, "CPU");
    auto infer_request = compiled_model.create_infer_request();
 
    auto input = model->input();
    std::string input_name = input.get_any_name();
 
    ov::Shape shape = {1};
 
    for (int64_t i = 0; i < 5; ++i) {
        ov::Tensor input_tensor(ov::element::i64, shape);
 
        int64_t* data = input_tensor.data<int64_t>();
        data[0] = i;
 
        infer_request.set_tensor(input_name, input_tensor);
 
        infer_request.infer();

        auto output_tensor = infer_request.get_output_tensor(0);
        int64_t* output_data = input_tensor.data<int64_t>();
        std::cout << output_data[0] << std::endl;
 
        std::cout << "Inference #" << i << " completed, input = " << i << std::endl;
    }
}

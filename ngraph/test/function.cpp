// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/function.hpp"

#include <gtest/gtest.h>

#include "openvino/core/partial_shape.hpp"
#include "openvino/opsets/opset8.hpp"

TEST(function, get_input_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input("input");
    ASSERT_EQ(input.get_node(), arg0.get());
    ASSERT_EQ(input.get_element_type(), ov::element::f32);
    ASSERT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_output_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    const std::unordered_set<std::string> out_names = {"relu_t", "identity"};
    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names(out_names);
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<ov::Function>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output("relu_t");
    ASSERT_EQ(output.get_tensor().get_names().size(), 2);
    ASSERT_EQ(output.get_tensor().get_names(), out_names);
    ASSERT_EQ(output.get_node(), result.get());
    ASSERT_EQ(f->output("identity"), output);
    ASSERT_EQ(output.get_element_type(), ov::element::f32);
    ASSERT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_incorrect_output_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    ASSERT_THROW(f->output("input"), ov::Exception);
}

TEST(function, get_incorrect_input_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    ASSERT_THROW(f->input("relu_t"), ov::Exception);
}

TEST(function, get_input_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input(0);
    ASSERT_EQ(input.get_node(), arg0.get());
    ASSERT_EQ(input.get_element_type(), ov::element::f32);
    ASSERT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_output_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<ov::Function>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output(0);
    ASSERT_EQ(output.get_node(), result.get());
    ASSERT_EQ(output.get_element_type(), ov::element::f32);
    ASSERT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_input_without_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input();
    ASSERT_EQ(input.get_node(), arg0.get());
    ASSERT_EQ(input.get_element_type(), ov::element::f32);
    ASSERT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_output_without_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<ov::Function>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output();
    ASSERT_EQ(output.get_node(), result.get());
    ASSERT_EQ(output.get_element_type(), ov::element::f32);
    ASSERT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_incorrect_output_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    ASSERT_THROW(f->output(2), std::exception);
}

TEST(function, get_incorrect_input_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    ASSERT_THROW(f->input(2), std::exception);
}

TEST(function, incorrect_multiple_inputs_outputs_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 3});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input1"});

    auto arg1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 3});
    arg1->set_friendly_name("data1");
    arg1->get_output_tensor(0).set_names({"input2", "data1"});

    auto concat = std::make_shared<ov::opset8::Concat>(ov::NodeVector{arg0, arg1}, 1);
    concat->set_friendly_name("concat");
    concat->get_output_tensor(0).set_names({"concat_t"});
    auto result1 = std::make_shared<ov::opset8::Result>(concat);

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result2 = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<ov::Function>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    f->validate_nodes_and_infer_types();

    ASSERT_THROW(f->input(), ov::Exception);
    ASSERT_THROW(f->output(), ov::Exception);
}

TEST(function, multiple_inputs_outputs_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 3, 3});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input1"});

    auto arg1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 3});
    arg1->set_friendly_name("data1");
    arg1->get_output_tensor(0).set_names({"input2", "data1"});

    auto concat = std::make_shared<ov::opset8::Concat>(ov::NodeVector{arg0, arg1}, 1);
    concat->set_friendly_name("concat");
    concat->get_output_tensor(0).set_names({"concat_t"});
    auto result1 = std::make_shared<ov::opset8::Result>(concat);

    auto shape_of = std::make_shared<ov::opset8::ShapeOf>(concat);
    shape_of->set_friendly_name("shape_of");
    shape_of->get_output_tensor(0).set_names({"shape_of_t", "identity"});
    auto result2 = std::make_shared<ov::opset8::Result>(shape_of);
    auto f = std::make_shared<ov::Function>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    f->validate_nodes_and_infer_types();

    auto input1 = f->input(0);
    auto input2 = f->input("data1");

    ASSERT_NE(input1, input2);
    ASSERT_EQ(input1, f->input("input1"));
    ASSERT_EQ(input2, f->input("input2"));
    ASSERT_EQ(input2, f->input(1));
    ASSERT_EQ(input1.get_node(), arg0.get());
    ASSERT_EQ(input2.get_node_shared_ptr(), arg1);

    auto output1 = f->output(0);
    auto output2 = f->output("shape_of_t");

    ASSERT_NE(output1, output2);
    ASSERT_EQ(output1, f->output("concat_t"));
    ASSERT_EQ(output2, f->output("identity"));
    ASSERT_EQ(output2, f->output(1));
    ASSERT_EQ(arg0.get(), f->input(0).get_node());
    ASSERT_EQ(arg1.get(), f->input(1).get_node());
    ASSERT_EQ(result1.get(), f->output(0).get_node());
    ASSERT_EQ(result2.get(), f->output(1).get_node());
    ASSERT_EQ(output1, result1);
    ASSERT_EQ(output2, result2);
    ASSERT_EQ(f->inputs().size(), 2);
    ASSERT_EQ(f->outputs().size(), 2);
}

TEST(function, DISABLED_create_function_with_incorrect_tensor_names) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"input"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});
    ASSERT_THROW(f->validate_nodes_and_infer_types(), ov::Exception);
}

TEST(function, get_input_by_tensor_name_from_const) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input("input");
    ASSERT_EQ(input.get_node(), arg0.get());
    ASSERT_EQ(input.get_element_type(), ov::element::f32);
    ASSERT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_output_by_tensor_name_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    const std::unordered_set<std::string> out_names = {"relu_t", "identity"};
    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names(out_names);
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<const ov::Function>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output("relu_t");
    ASSERT_EQ(output.get_tensor().get_names().size(), 2);
    ASSERT_EQ(output.get_tensor().get_names(), out_names);
    ASSERT_EQ(output.get_node(), result.get());
    ASSERT_EQ(f->output("identity"), output);
    ASSERT_EQ(output.get_element_type(), ov::element::f32);
    ASSERT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_incorrect_output_by_tensor_name_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    ASSERT_THROW(f->output("input"), ov::Exception);
}

TEST(function, get_incorrect_input_by_tensor_name_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    ASSERT_THROW(f->input("relu_t"), ov::Exception);
}

TEST(function, get_input_by_index_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input(0);
    ASSERT_EQ(input.get_node(), arg0.get());
    ASSERT_EQ(input.get_element_type(), ov::element::f32);
    ASSERT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_output_by_index_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<const ov::Function>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output(0);
    ASSERT_EQ(output.get_node(), result.get());
    ASSERT_EQ(output.get_element_type(), ov::element::f32);
    ASSERT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_input_without_index_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input();
    ASSERT_EQ(input.get_node(), arg0.get());
    ASSERT_EQ(input.get_element_type(), ov::element::f32);
    ASSERT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_output_without_index_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<const ov::Function>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output();
    ASSERT_EQ(output.get_node(), result.get());
    ASSERT_EQ(output.get_element_type(), ov::element::f32);
    ASSERT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(function, get_incorrect_output_by_index_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    ASSERT_THROW(f->output(2), std::exception);
}

TEST(function, get_incorrect_input_by_index_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Function>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    ASSERT_THROW(f->input(2), std::exception);
}

TEST(function, incorrect_multiple_inputs_outputs_function_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 3});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input1"});

    auto arg1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 3});
    arg1->set_friendly_name("data1");
    arg1->get_output_tensor(0).set_names({"input2", "data1"});

    auto concat = std::make_shared<ov::opset8::Concat>(ov::NodeVector{arg0, arg1}, 1);
    concat->set_friendly_name("concat");
    concat->get_output_tensor(0).set_names({"concat_t"});
    auto result1 = std::make_shared<ov::opset8::Result>(concat);

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result2 = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<const ov::Function>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    f->validate_nodes_and_infer_types();

    ASSERT_THROW(f->input(), ov::Exception);
    ASSERT_THROW(f->output(), ov::Exception);
}

TEST(function, multiple_inputs_outputs_function_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 3, 3});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input1"});

    auto arg1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 3});
    arg1->set_friendly_name("data1");
    arg1->get_output_tensor(0).set_names({"input2", "data1"});

    auto concat = std::make_shared<ov::opset8::Concat>(ov::NodeVector{arg0, arg1}, 1);
    concat->set_friendly_name("concat");
    concat->get_output_tensor(0).set_names({"concat_t"});
    auto result1 = std::make_shared<ov::opset8::Result>(concat);

    auto shape_of = std::make_shared<ov::opset8::ShapeOf>(concat);
    shape_of->set_friendly_name("shape_of");
    shape_of->get_output_tensor(0).set_names({"shape_of_t", "identity"});
    auto result2 = std::make_shared<ov::opset8::Result>(shape_of);
    auto f = std::make_shared<const ov::Function>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    f->validate_nodes_and_infer_types();

    auto input1 = f->input(0);
    auto input2 = f->input("data1");

    ASSERT_NE(input1, input2);
    ASSERT_EQ(input1, f->input("input1"));
    ASSERT_EQ(input2, f->input("input2"));
    ASSERT_EQ(input2, f->input(1));
    ASSERT_EQ(input1.get_node(), arg0.get());
    ASSERT_EQ(input2.get_node_shared_ptr(), arg1);

    auto output1 = f->output(0);
    auto output2 = f->output("shape_of_t");

    ASSERT_NE(output1, output2);
    ASSERT_EQ(output1, f->output("concat_t"));
    ASSERT_EQ(output2, f->output("identity"));
    ASSERT_EQ(arg0.get(), f->input(0).get_node());
    ASSERT_EQ(arg1.get(), f->input(1).get_node());
    ASSERT_EQ(result1.get(), f->output(0).get_node());
    ASSERT_EQ(result2.get(), f->output(1).get_node());
    ASSERT_EQ(output2, f->output(1));
    ASSERT_EQ(output1.get_node(), result1.get());
    ASSERT_EQ(output2.get_node(), result2.get());
    ASSERT_EQ(f->inputs().size(), 2);
    ASSERT_EQ(f->outputs().size(), 2);
}

TEST(function, DISABLED_create_function_with_incorrect_tensor_names_from_const_function) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"input"});
    auto f = std::make_shared<const ov::Function>(relu, ov::ParameterVector{arg0});
    ASSERT_THROW(f->validate_nodes_and_infer_types(), ov::Exception);
}

TEST(function_reshape, ReshapedDynamicShapeLayout) {
    std::shared_ptr<ov::Function> ngraph;
    {
        ov::PartialShape shape({-1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor", "tensor2"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);

        ov::ParameterVector params = {param};
        ngraph = std::make_shared<ov::Function>(relu, params);
    }

    EXPECT_TRUE(ngraph->input().get_partial_shape().is_dynamic());

    std::map<std::string, ov::PartialShape> new_shape;
    new_shape["tensor"] = ov::Shape{1, 3, 22, 22};
    ASSERT_NO_THROW(ngraph->reshape(new_shape));

    EXPECT_FALSE(ngraph->input().get_partial_shape().is_dynamic());
    EXPECT_FALSE(ngraph->get_parameters().front()->get_partial_shape().is_dynamic());
}

TEST(function_reshape, ReshapeBatchReLU) {
    std::shared_ptr<ov::Function> ngraph;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor", "tensor2"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        ngraph = std::make_shared<ov::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        std::map<std::string, ov::PartialShape> new_shape;
        new_shape["tensor2"] = ov::PartialShape{2, 3, 22, 22};
        ASSERT_NO_THROW(ngraph->reshape(new_shape));
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ov::Shape({2, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ov::Shape({2, 3, 22, 22}));
}

TEST(function_reshape, ReshapeSpatialReLU) {
    std::shared_ptr<ov::Function> ngraph;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        ngraph = std::make_shared<ov::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        std::map<std::string, ov::PartialShape> new_shape;
        new_shape["tensor"] = ov::PartialShape{1, 3, 25, 25};
        ASSERT_NO_THROW(ngraph->reshape(new_shape));
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 25, 25}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ov::Shape({1, 3, 25, 25}));
}

TEST(function_reshape, ReshapeSpatialReLUWithoutReplaceParameter) {
    std::shared_ptr<ov::Function> ngraph;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        ngraph = std::make_shared<ov::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        ngraph->get_parameters()[0]->set_partial_shape({1, 3, 25, 25});
        ngraph->validate_nodes_and_infer_types();
    }

    ASSERT_EQ(ngraph->input().get_partial_shape(), ov::Shape({1, 3, 25, 25}));
    ASSERT_EQ(ngraph->output().get_partial_shape(), ov::Shape({1, 3, 25, 25}));
}

TEST(function_reshape, ReshapeSpatialReLUStaticToDynamic) {
    const ov::PartialShape refShape{1, 3, ov::Dimension::dynamic(), 25};
    std::shared_ptr<ov::Function> ngraph;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        ngraph = std::make_shared<ov::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        std::map<std::string, ov::PartialShape> new_shape;
        new_shape["tensor"] = refShape;
        ASSERT_NO_THROW(ngraph->reshape(new_shape));
    }

    ASSERT_TRUE(ngraph->input(0).get_partial_shape().is_dynamic());
    ASSERT_TRUE(ngraph->output(0).get_partial_shape().is_dynamic());
    ASSERT_EQ(ngraph->input(0).get_partial_shape(), refShape);
    ASSERT_EQ(ngraph->output(0).get_partial_shape(), refShape);
}

TEST(function_reshape, ReshapeSpatialReLUStaticToFullyDynamic) {
    const ov::PartialShape refShape = ov::PartialShape::dynamic();
    std::shared_ptr<ov::Function> ngraph;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        ngraph = std::make_shared<ov::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        std::map<std::string, ov::PartialShape> new_shape;
        new_shape["tensor"] = refShape;
        ASSERT_NO_THROW(ngraph->reshape(new_shape));
    }

    ASSERT_TRUE(ngraph->input().get_partial_shape().is_dynamic());
    ASSERT_TRUE(ngraph->output().get_partial_shape().is_dynamic());
    ASSERT_EQ(ngraph->input().get_partial_shape(), refShape);
    ASSERT_EQ(ngraph->output().get_partial_shape(), refShape);
}

TEST(function_reshape, ReshapeSpatialReLUDynamicToDynamic) {
    const ov::PartialShape refShape{1, 3, ov::Dimension::dynamic(), 25};
    std::shared_ptr<ov::Function> ngraph;
    {
        ov::PartialShape shape({1, 3, 22, ov::Dimension::dynamic()});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        ngraph = std::make_shared<ov::Function>(results, params);
    }

    ASSERT_EQ(ngraph->input().get_partial_shape(), ov::PartialShape({1, 3, 22, ov::Dimension::dynamic()}));
    ASSERT_EQ(ngraph->output().get_partial_shape(), ov::PartialShape({1, 3, 22, ov::Dimension::dynamic()}));

    {
        std::map<std::string, ov::PartialShape> new_shape;
        new_shape["tensor"] = refShape;
        ASSERT_NO_THROW(ngraph->reshape(new_shape));
    }

    ASSERT_TRUE(ngraph->input().get_partial_shape().is_dynamic());
    ASSERT_TRUE(ngraph->output().get_partial_shape().is_dynamic());
    ASSERT_EQ(ngraph->input().get_partial_shape(), refShape);
    ASSERT_EQ(ngraph->output().get_partial_shape(), refShape);
}

TEST(function_reshape, TestInvalidReshape) {
    std::shared_ptr<ov::Function> f;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1000, 4});
        input->get_output_tensor(0).set_names({"tensor"});
        auto shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 4000});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input, shape, true);
        f = std::make_shared<ov::Function>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }

    ASSERT_ANY_THROW(f->reshape({{"tensor", ov::Shape({4})}}));

    auto param = f->get_parameters().front();
    ASSERT_EQ(param->get_output_shape(0), ov::Shape({1, 1000, 4}));

    ASSERT_NO_THROW(f->reshape({{"tensor", ov::Shape({1, 1000, 4})}}));
}

TEST(function_reshape, TestReshapeWithInvalidTensorName) {
    std::shared_ptr<ov::Function> f;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1000, 4});
        input->set_friendly_name("param");
        input->get_output_tensor(0).set_names({"tensor"});
        auto shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 4000});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input, shape, true);
        f = std::make_shared<ov::Function>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }

    // both operation names and tensor names are specified
    ASSERT_ANY_THROW(f->reshape({{"param", ov::Shape({4, 4, 4})}, {"tensor", ov::Shape({4, 4, 4})}}));

    // operation name does not work
    ASSERT_ANY_THROW(f->reshape({{"param", ov::Shape({4, 4, 4})}}));
}

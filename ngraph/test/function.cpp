// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/function.hpp"

#include <gtest/gtest.h>

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
    ASSERT_EQ(output1, result1);
    ASSERT_EQ(output2, result2);
}

TEST(function, create_function_with_incorrect_tensor_names) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"input"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});
    ASSERT_THROW(f->validate_nodes_and_infer_types(), ov::Exception);
}

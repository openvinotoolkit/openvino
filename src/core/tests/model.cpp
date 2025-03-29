// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/model.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/opsets/opset8.hpp"
#include "shared_node_info.hpp"

using ov::op::util::Variable;
using ov::op::util::VariableInfo;
using ov::op::v0::Parameter;
using ov::op::v0::Result;

TEST(model, get_input_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input("input");
    EXPECT_EQ(input.get_node(), arg0.get());
    EXPECT_EQ(input.get_element_type(), ov::element::f32);
    EXPECT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_output_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    const std::unordered_set<std::string> out_names = {"relu_t", "identity"};
    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names(out_names);
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<ov::Model>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output("relu_t");
    EXPECT_EQ(output.get_tensor().get_names().size(), 2);
    EXPECT_EQ(output.get_tensor().get_names(), out_names);
    EXPECT_EQ(output.get_node(), result.get());
    EXPECT_EQ(f->output("identity"), output);
    EXPECT_EQ(output.get_element_type(), ov::element::f32);
    EXPECT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_input_by_tensor_index_without_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    auto f = std::make_shared<ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input(0);
    EXPECT_THROW(f->input("input"), ov::Exception);
    EXPECT_EQ(input.get_node(), arg0.get());
    EXPECT_EQ(input.get_element_type(), ov::element::f32);
    EXPECT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_output_by_tensor_index_without_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<ov::Model>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output(0);
    EXPECT_THROW(f->output("relu_t"), ov::Exception);
    EXPECT_EQ(output.get_tensor().get_names().size(), 0);
    EXPECT_EQ(output.get_node(), result.get());
    EXPECT_THROW(f->output("identity"), ov::Exception);
    EXPECT_EQ(output.get_element_type(), ov::element::f32);
    EXPECT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_incorrect_output_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_THROW(f->output("input"), ov::Exception);
}

TEST(model, get_incorrect_input_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_THROW(f->input("relu_t"), ov::Exception);
}

TEST(model, get_input_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input(0);
    EXPECT_EQ(input.get_node(), arg0.get());
    EXPECT_EQ(input.get_element_type(), ov::element::f32);
    EXPECT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_output_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<ov::Model>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output(0);
    EXPECT_EQ(output.get_node(), result.get());
    EXPECT_EQ(output.get_element_type(), ov::element::f32);
    EXPECT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_input_without_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input();
    EXPECT_EQ(input.get_node(), arg0.get());
    EXPECT_EQ(input.get_element_type(), ov::element::f32);
    EXPECT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_output_without_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<ov::Model>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output();
    EXPECT_EQ(output.get_node(), result.get());
    EXPECT_EQ(output.get_element_type(), ov::element::f32);
    EXPECT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_incorrect_output_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_THROW(f->output(2), std::exception);
}

TEST(model, get_incorrect_input_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_THROW(f->input(2), std::exception);
}

TEST(model, incorrect_multiple_inputs_outputs_model) {
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
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    f->validate_nodes_and_infer_types();

    EXPECT_THROW(f->input(), ov::Exception);
    EXPECT_THROW(f->output(), ov::Exception);
}

TEST(model, multiple_inputs_outputs_model) {
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
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    f->validate_nodes_and_infer_types();

    auto input1 = f->input(0);
    auto input2 = f->input("data1");

    EXPECT_NE(input1, input2);
    EXPECT_EQ(input1, f->input("input1"));
    EXPECT_EQ(input2, f->input("input2"));
    EXPECT_EQ(input2, f->input(1));
    EXPECT_EQ(input1.get_node(), arg0.get());
    EXPECT_EQ(input2.get_node_shared_ptr(), arg1);

    auto output1 = f->output(0);
    auto output2 = f->output("shape_of_t");

    EXPECT_NE(output1, output2);
    EXPECT_EQ(output1, f->output("concat_t"));
    EXPECT_EQ(output2, f->output("identity"));
    EXPECT_EQ(output2, f->output(1));
    EXPECT_EQ(arg0.get(), f->input(0).get_node());
    EXPECT_EQ(arg1.get(), f->input(1).get_node());
    EXPECT_EQ(result1.get(), f->output(0).get_node());
    EXPECT_EQ(result2.get(), f->output(1).get_node());
    EXPECT_EQ(output1, result1);
    EXPECT_EQ(output2, result2);
    EXPECT_EQ(f->inputs().size(), 2);
    EXPECT_EQ(f->outputs().size(), 2);
}

TEST(model, get_input_by_tensor_name_from_const) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input("input");
    EXPECT_EQ(input.get_node(), arg0.get());
    EXPECT_EQ(input.get_element_type(), ov::element::f32);
    EXPECT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_output_by_tensor_name_from_const_model) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    const std::unordered_set<std::string> out_names = {"relu_t", "identity"};
    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names(out_names);
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<const ov::Model>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output("relu_t");
    EXPECT_EQ(output.get_tensor().get_names().size(), 2);
    EXPECT_EQ(output.get_tensor().get_names(), out_names);
    EXPECT_EQ(output.get_node(), result.get());
    EXPECT_EQ(f->output("identity"), output);
    EXPECT_EQ(output.get_element_type(), ov::element::f32);
    EXPECT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_incorrect_output_by_tensor_name_from_const_model) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_THROW(f->output("input"), ov::Exception);
}

TEST(model, get_incorrect_input_by_tensor_name_from_const_model) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_THROW(f->input("relu_t"), ov::Exception);
}

TEST(model, get_input_by_index_from_const_model) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input(0);
    EXPECT_EQ(input.get_node(), arg0.get());
    EXPECT_EQ(input.get_element_type(), ov::element::f32);
    EXPECT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_output_by_index_from_const_model) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<const ov::Model>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output(0);
    EXPECT_EQ(output.get_node(), result.get());
    EXPECT_EQ(output.get_element_type(), ov::element::f32);
    EXPECT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_input_without_index_from_const_model) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto input = f->input();
    EXPECT_EQ(input.get_node(), arg0.get());
    EXPECT_EQ(input.get_element_type(), ov::element::f32);
    EXPECT_EQ(input.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_output_without_index_from_const_model) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<const ov::Model>(result, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    auto output = f->output();
    EXPECT_EQ(output.get_node(), result.get());
    EXPECT_EQ(output.get_element_type(), ov::element::f32);
    EXPECT_EQ(output.get_partial_shape(), ov::PartialShape{1});
}

TEST(model, get_incorrect_output_by_index_from_const_model) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_THROW(f->output(2), std::exception);
}

TEST(model, get_incorrect_input_by_index_from_const_model) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<const ov::Model>(relu, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_THROW(f->input(2), std::exception);
}

TEST(model, incorrect_multiple_inputs_outputs_model_from_const_model) {
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
    auto f = std::make_shared<const ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    f->validate_nodes_and_infer_types();

    EXPECT_THROW(f->input(), ov::Exception);
    EXPECT_THROW(f->output(), ov::Exception);
}

TEST(model, multiple_inputs_outputs_model_from_const_model) {
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
    auto f = std::make_shared<const ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    f->validate_nodes_and_infer_types();

    auto input1 = f->input(0);
    auto input2 = f->input("data1");

    EXPECT_NE(input1, input2);
    EXPECT_EQ(input1, f->input("input1"));
    EXPECT_EQ(input2, f->input("input2"));
    EXPECT_EQ(input2, f->input(1));
    EXPECT_EQ(input1.get_node(), arg0.get());
    EXPECT_EQ(input2.get_node_shared_ptr(), arg1);

    auto output1 = f->output(0);
    auto output2 = f->output("shape_of_t");

    EXPECT_NE(output1, output2);
    EXPECT_EQ(output1, f->output("concat_t"));
    EXPECT_EQ(output2, f->output("identity"));
    EXPECT_EQ(arg0.get(), f->input(0).get_node());
    EXPECT_EQ(arg1.get(), f->input(1).get_node());
    EXPECT_EQ(result1.get(), f->output(0).get_node());
    EXPECT_EQ(result2.get(), f->output(1).get_node());
    EXPECT_EQ(output2, f->output(1));
    EXPECT_EQ(output1.get_node(), result1.get());
    EXPECT_EQ(output2.get_node(), result2.get());
    EXPECT_EQ(f->inputs().size(), 2);
    EXPECT_EQ(f->outputs().size(), 2);
}

TEST(model, parameter_result_function) {
    std::shared_ptr<ov::Model> function = nullptr;
    {
        auto param = std::make_shared<ov::opset8::Parameter>(ov::element::f16, ov::Shape({1, 3, 24, 24}));
        param->set_friendly_name("param");
        param->output(0).get_tensor().set_names({"data"});
        auto result = std::make_shared<ov::opset8::Result>(param);
        result->set_friendly_name("result");
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
        function->set_friendly_name("ParamResult");
    }

    EXPECT_EQ(function->inputs().size(), 1);
    EXPECT_NO_THROW(function->input());
    EXPECT_NO_THROW(function->input("data"));
    EXPECT_THROW(function->input("param"), ov::Exception);

    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_NO_THROW(function->output());
    EXPECT_EQ(1, function->output(0).get_tensor().get_names().size());
    EXPECT_NO_THROW(function->output("data"));
    EXPECT_THROW(function->output("constant"), ov::Exception);

    EXPECT_EQ(ov::element::f16, function->input("data").get_element_type());
    EXPECT_EQ(ov::element::f16, function->output("data").get_element_type());
}

TEST(model, constant_result_function) {
    std::shared_ptr<ov::Model> function = nullptr;
    std::shared_ptr<ov::Node> constant = nullptr;

    {
        constant = std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape({1, 3, 24, 24}));
        constant->set_friendly_name("constant");
        constant->output(0).get_tensor().set_names({"data"});
        auto result = std::make_shared<ov::opset8::Result>(constant);
        result->set_friendly_name("result");
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{});
        function->set_friendly_name("ConstResult");
    }

    EXPECT_EQ(function->inputs().size(), 0);
    EXPECT_THROW(function->input(), ov::Exception);
    EXPECT_THROW(function->input("data"), ov::Exception);
    EXPECT_THROW(function->input("constant"), ov::Exception);

    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_NO_THROW(function->output());
    EXPECT_EQ(1, function->output(0).get_tensor().get_names().size());
    EXPECT_NO_THROW(function->output("data"));
    EXPECT_THROW(function->output("constant"), ov::Exception);
    EXPECT_EQ(ov::element::f32, function->output("data").get_element_type());
}

TEST(model_reshape, ReshapedDynamicShapeLayout) {
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({-1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor", "tensor2"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);

        ov::ParameterVector params = {param};
        model = std::make_shared<ov::Model>(relu, params);
    }

    EXPECT_TRUE(model->input().get_partial_shape().is_dynamic());

    std::map<std::string, ov::PartialShape> new_shape;
    new_shape["tensor"] = ov::Shape{1, 3, 22, 22};
    EXPECT_NO_THROW(model->reshape(new_shape));

    EXPECT_FALSE(model->input().get_partial_shape().is_dynamic());
    EXPECT_FALSE(model->get_parameters().front()->get_partial_shape().is_dynamic());
}

TEST(model_reshape, ReshapeBatchReLU) {
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor", "tensor2"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        model = std::make_shared<ov::Model>(results, params);
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        std::map<std::string, ov::PartialShape> new_shape;
        new_shape["tensor2"] = ov::PartialShape{2, 3, 22, 22};
        EXPECT_NO_THROW(model->reshape(new_shape));
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({2, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({2, 3, 22, 22}));
}

TEST(model_reshape, ReshapeSpatialReLU) {
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        model = std::make_shared<ov::Model>(results, params);
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        std::map<std::string, ov::PartialShape> new_shape;
        new_shape["tensor"] = ov::PartialShape{1, 3, 25, 25};
        EXPECT_NO_THROW(model->reshape(new_shape));
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 25, 25}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 3, 25, 25}));
}

TEST(model_reshape, ReshapeSpatialReLUWithoutReplaceParameter) {
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        model = std::make_shared<ov::Model>(results, params);
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        model->get_parameters()[0]->set_partial_shape({1, 3, 25, 25});
        model->validate_nodes_and_infer_types();
    }

    EXPECT_EQ(model->input().get_partial_shape(), ov::Shape({1, 3, 25, 25}));
    EXPECT_EQ(model->output().get_partial_shape(), ov::Shape({1, 3, 25, 25}));
}

TEST(model_reshape, ReshapeSpatialReLUStaticToDynamic) {
    const ov::PartialShape refShape{1, 3, ov::Dimension::dynamic(), 25};
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        model = std::make_shared<ov::Model>(results, params);
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        std::map<std::string, ov::PartialShape> new_shape;
        new_shape["tensor"] = refShape;
        EXPECT_NO_THROW(model->reshape(new_shape));
    }

    EXPECT_TRUE(model->input(0).get_partial_shape().is_dynamic());
    EXPECT_TRUE(model->output(0).get_partial_shape().is_dynamic());
    EXPECT_EQ(model->input(0).get_partial_shape(), refShape);
    EXPECT_EQ(model->output(0).get_partial_shape(), refShape);
}

TEST(model_reshape, ReshapeSpatialReLUStaticToFullyDynamic) {
    const ov::PartialShape refShape = ov::PartialShape::dynamic();
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        model = std::make_shared<ov::Model>(results, params);
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        std::map<std::string, ov::PartialShape> new_shape;
        new_shape["tensor"] = refShape;
        EXPECT_NO_THROW(model->reshape(new_shape));
    }

    EXPECT_TRUE(model->input().get_partial_shape().is_dynamic());
    EXPECT_TRUE(model->output().get_partial_shape().is_dynamic());
    EXPECT_EQ(model->input().get_partial_shape(), refShape);
    EXPECT_EQ(model->output().get_partial_shape(), refShape);
}

TEST(model_reshape, ReshapeSpatialReLUDynamicToDynamic) {
    const ov::PartialShape refShape{1, 3, ov::Dimension::dynamic(), 25};
    std::shared_ptr<ov::Model> model;
    {
        ov::PartialShape shape({1, 3, 22, ov::Dimension::dynamic()});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor"});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        model = std::make_shared<ov::Model>(results, params);
    }

    EXPECT_EQ(model->input().get_partial_shape(), ov::PartialShape({1, 3, 22, ov::Dimension::dynamic()}));
    EXPECT_EQ(model->output().get_partial_shape(), ov::PartialShape({1, 3, 22, ov::Dimension::dynamic()}));

    {
        std::map<std::string, ov::PartialShape> new_shape;
        new_shape["tensor"] = refShape;
        EXPECT_NO_THROW(model->reshape(new_shape));
    }

    EXPECT_TRUE(model->input().get_partial_shape().is_dynamic());
    EXPECT_TRUE(model->output().get_partial_shape().is_dynamic());
    EXPECT_EQ(model->input().get_partial_shape(), refShape);
    EXPECT_EQ(model->output().get_partial_shape(), refShape);
}

TEST(model_reshape, TestInvalidReshape) {
    std::shared_ptr<ov::Model> f;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1000, 4});
        input->get_output_tensor(0).set_names({"tensor"});
        auto shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 4000});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input, shape, true);
        f = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }

    EXPECT_THROW(f->reshape({{"tensor", ov::Shape({4})}}), ov::Exception);

    auto param = f->get_parameters().front();
    EXPECT_EQ(param->get_output_shape(0), ov::Shape({1, 1000, 4}));

    EXPECT_NO_THROW(f->reshape({{"tensor", ov::Shape({1, 1000, 4})}}));
}

TEST(model_reshape, TestReshapeWithInvalidTensorName) {
    std::shared_ptr<ov::Model> f;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1000, 4});
        input->set_friendly_name("param");
        input->get_output_tensor(0).set_names({"tensor"});
        auto shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 4000});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input, shape, true);
        f = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }

    // both operation names and tensor names are specified
    EXPECT_ANY_THROW(f->reshape({{"param", ov::Shape({4, 4, 4})}, {"tensor", ov::Shape({4, 4, 4})}}));

    // operation name does not work
    EXPECT_ANY_THROW(f->reshape({{"param", ov::Shape({4, 4, 4})}}));
}

TEST(model_reshape, TestReshapeWithInvalidShapesForTheSameTensor) {
    std::shared_ptr<ov::Model> f;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1000, 4});
        input->set_friendly_name("param");
        input->get_output_tensor(0).set_names({"tensor1", "tensor2"});
        auto shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 4000});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input, shape, true);
        f = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }

    // both tensor names are specified, but have different shapes
    EXPECT_ANY_THROW(f->reshape({{"tensor1", ov::Shape({2, 500, 4})}, {"tensor2", ov::Shape({4, 250, 4})}}));
}

TEST(model_reshape, ReshapeBatchReLUByIndex) {
    std::shared_ptr<ov::Model> model;
    ov::Output<ov::Node> port;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor", "tensor2"});
        port = param->output(0);
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        model = std::make_shared<ov::Model>(results, params);
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        std::map<size_t, ov::PartialShape> new_shape;
        new_shape[0] = ov::PartialShape{2, 3, 22, 22};
        EXPECT_NO_THROW(model->reshape(new_shape));
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({2, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({2, 3, 22, 22}));
}

TEST(model_reshape, ReshapeBatchReLUByPort) {
    std::shared_ptr<ov::Model> model;
    ov::Output<ov::Node> port;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor", "tensor2"});
        port = param->output(0);
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        model = std::make_shared<ov::Model>(results, params);
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        std::map<ov::Output<ov::Node>, ov::PartialShape> new_shape;
        new_shape[port] = ov::PartialShape{2, 3, 22, 22};
        EXPECT_NO_THROW(model->reshape(new_shape));
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({2, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({2, 3, 22, 22}));
}

TEST(model_reshape, ReshapeBatchReLUWithOneInput) {
    std::shared_ptr<ov::Model> model;
    ov::Output<ov::Node> port;
    {
        ov::PartialShape shape({1, 3, 22, 22});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->get_output_tensor(0).set_names({"tensor", "tensor2"});
        port = param->output(0);
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        model = std::make_shared<ov::Model>(results, params);
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 3, 22, 22}));

    {
        ov::PartialShape new_shape;
        new_shape = ov::PartialShape{2, 3, 22, 22};
        EXPECT_NO_THROW(model->reshape(new_shape));
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({2, 3, 22, 22}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({2, 3, 22, 22}));
}

TEST(model_reshape, IncorrectReshapeBatchWithMultipleInputs) {
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
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    f->validate_nodes_and_infer_types();
    ov::PartialShape shape({1, 3, 22, 22});
    EXPECT_THROW(f->reshape(shape), ov::Exception);
}

TEST(model_reshape, ReshapeWithStaticVariableSingleInput) {
    auto arg0 = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 3, 4, 5});

    auto variable = std::make_shared<Variable>(VariableInfo{arg0->get_output_partial_shape(0), ov::element::f32, "ID"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(arg0, variable);
    auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);

    auto result1 = std::make_shared<Result>(assign);

    auto shape_of = std::make_shared<ov::opset8::ShapeOf>(arg0);
    auto result2 = std::make_shared<Result>(shape_of);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0});

    model->validate_nodes_and_infer_types();

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 4, 5}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 3, 4, 5}));
    EXPECT_EQ(model->get_results()[1]->get_shape(), ov::Shape({4}));

    {
        ov::PartialShape shape({1, 4, 3, 3});
        model->reshape(shape, {{"ID", shape}});
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 4, 3, 3}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 4, 3, 3}));
    EXPECT_EQ(model->get_results()[1]->get_shape(), ov::Shape({4}));
}

TEST(model_reshape, ReshapeWithStaticVariablesSingleInput) {
    auto arg0 = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 3, 4, 5});

    auto variable =
        std::make_shared<Variable>(VariableInfo{arg0->get_output_partial_shape(0), ov::element::f32, "ID1"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(arg0, variable);
    auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);

    auto add = std::make_shared<ov::op::v1::Add>(read_value, read_value);
    auto var_add = std::make_shared<Variable>(VariableInfo{add->get_output_partial_shape(0), ov::element::f32, "ID2"});
    auto read_value_add = std::make_shared<ov::op::v6::ReadValue>(add, var_add);
    auto assign_add = std::make_shared<ov::op::v6::Assign>(read_value_add, var_add);

    auto result1 = std::make_shared<Result>(assign);

    auto shape_of = std::make_shared<ov::opset8::ShapeOf>(add);
    auto result2 = std::make_shared<Result>(shape_of);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0});

    model->validate_nodes_and_infer_types();

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 4, 5}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 3, 4, 5}));
    EXPECT_EQ(model->get_results()[1]->get_shape(), ov::Shape({4}));

    {
        ov::PartialShape shape({1, 4, 3, 3});
        model->reshape(shape, {{"ID2", shape}, {"ID1", shape}});
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 4, 3, 3}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 4, 3, 3}));
    EXPECT_EQ(model->get_results()[1]->get_shape(), ov::Shape({4}));
}

TEST(model_reshape, ReshapeWithDynamicVariableSingleInput) {
    auto arg0 = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 3, 4, 5});
    arg0->get_output_tensor(0).set_names({"input"});

    auto variable = std::make_shared<Variable>(VariableInfo{ov::PartialShape::dynamic(4), ov::element::f32, "ID"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(arg0, variable);
    auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);

    auto result1 = std::make_shared<Result>(assign);

    auto shape_of = std::make_shared<ov::opset8::ShapeOf>(arg0);
    auto result2 = std::make_shared<Result>(shape_of);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0});

    model->validate_nodes_and_infer_types();

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 4, 5}));
    EXPECT_EQ(model->get_results()[0]->get_output_partial_shape(0), ov::PartialShape::dynamic(4));
    EXPECT_EQ(model->get_results()[1]->get_output_partial_shape(0), ov::Shape({4}));

    model->reshape({{"input", ov::PartialShape{1, 4, 8, 9}}});

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 4, 8, 9}));
    EXPECT_EQ(model->get_results()[0]->get_output_partial_shape(0), ov::PartialShape::dynamic(4));
    EXPECT_EQ(model->get_results()[1]->get_shape(), ov::Shape({4}));
}

TEST(model_reshape, ReshapeStaticWithVariableMultipleInputs) {
    auto arg0 = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 3, 4, 5});
    auto arg1 = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 2, 4, 5});
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{arg0, arg1}, 1);

    auto variable =
        std::make_shared<Variable>(VariableInfo{concat->get_output_partial_shape(0), ov::element::f32, "ID"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(concat, variable);
    auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);

    auto result1 = std::make_shared<Result>(assign);

    auto shape_of = std::make_shared<ov::opset8::ShapeOf>(concat);
    auto result2 = std::make_shared<Result>(shape_of);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    model->validate_nodes_and_infer_types();

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 4, 5}));
    EXPECT_EQ(model->get_parameters()[1]->get_shape(), ov::Shape({1, 2, 4, 5}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 5, 4, 5}));
    EXPECT_EQ(model->get_results()[1]->get_shape(), ov::Shape({4}));

    {
        ov::PartialShape shape({1, 14, 4, 5});
        model->reshape({{0, shape}, {1, shape}}, {{"ID", {1, 28, 4, 5}}});
    }

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 14, 4, 5}));
    EXPECT_EQ(model->get_parameters()[1]->get_shape(), ov::Shape({1, 14, 4, 5}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 28, 4, 5}));
    EXPECT_EQ(model->get_results()[1]->get_shape(), ov::Shape({4}));
}

TEST(model_reshape, ReshapeWithStaticVariableIncorrectVariable) {
    auto arg0 = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 3, 4, 5});
    auto arg1 = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 2, 4, 5});
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{arg0, arg1}, 1);

    auto variable =
        std::make_shared<Variable>(VariableInfo{concat->get_output_partial_shape(0), ov::element::f32, "ID"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(concat, variable);
    auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);

    auto result1 = std::make_shared<Result>(assign);

    auto shape_of = std::make_shared<ov::opset8::ShapeOf>(concat);
    auto result2 = std::make_shared<Result>(shape_of);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    model->validate_nodes_and_infer_types();

    EXPECT_EQ(model->get_parameters()[0]->get_shape(), ov::Shape({1, 3, 4, 5}));
    EXPECT_EQ(model->get_parameters()[1]->get_shape(), ov::Shape({1, 2, 4, 5}));
    EXPECT_EQ(model->get_results()[0]->get_shape(), ov::Shape({1, 5, 4, 5}));
    EXPECT_EQ(model->get_results()[1]->get_shape(), ov::Shape({4}));

    ov::PartialShape shape({1, 14, 4, 5});
    ov::PartialShape wrong_var_shape({1, 20, 4, 5});
    EXPECT_THROW(model->reshape({{0, shape}, {1, shape}}), ov::AssertFailure);
    EXPECT_THROW(model->reshape({{0, shape}, {1, shape}}, {{"WRONG_ID", {1, 28, 4, 5}}}), ov::AssertFailure);
    EXPECT_THROW(model->reshape({{0, shape}, {1, shape}}, {{"ID", wrong_var_shape}}), ov::AssertFailure);
}

TEST(model, add_output_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    relu1->set_friendly_name("relu1");
    relu1->get_output_tensor(0).set_names({"relu_t1"});

    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    relu2->set_friendly_name("relu2");
    relu2->get_output_tensor(0).set_names({"relu_t2"});
    auto f = std::make_shared<ov::Model>(relu2, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_EQ(f->get_results().size(), 1);

    ov::Output<ov::Node> out1, out2;
    EXPECT_NO_THROW(out1 = f->add_output("relu_t1"));
    EXPECT_EQ(f->get_results().size(), 2);
    EXPECT_NO_THROW(out2 = f->add_output("relu_t1"));
    EXPECT_EQ(f->get_results().size(), 2);
    EXPECT_EQ(f->get_results()[1]->input_value(0).get_node(), relu1.get());
    EXPECT_EQ(out1, out2);
    EXPECT_EQ(out1.get_node(), f->get_results()[1].get());
}

TEST(model, add_output_op_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    relu1->set_friendly_name("relu1");
    relu1->get_output_tensor(0).set_names({"relu_t1"});

    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    relu2->set_friendly_name("relu2");
    relu2->get_output_tensor(0).set_names({"relu_t2"});
    auto f = std::make_shared<ov::Model>(relu2, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_EQ(f->get_results().size(), 1);

    EXPECT_NO_THROW(f->add_output("relu1", 0));
    EXPECT_EQ(f->get_results().size(), 2);
    EXPECT_NO_THROW(f->add_output("relu_t1"));
    EXPECT_EQ(f->get_results().size(), 2);
    EXPECT_NO_THROW(f->add_output("relu2", 0));
    EXPECT_EQ(f->get_results().size(), 2);
    EXPECT_EQ(f->get_results()[1]->input_value(0).get_node(), relu1.get());
}

TEST(model, add_output_port) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    relu1->set_friendly_name("relu1");
    relu1->get_output_tensor(0).set_names({"relu_t1"});

    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    relu2->set_friendly_name("relu2");
    relu2->get_output_tensor(0).set_names({"relu_t2"});
    auto f = std::make_shared<ov::Model>(relu2, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_EQ(f->get_results().size(), 1);

    ov::Output<ov::Node> out;
    EXPECT_NO_THROW(out = f->add_output(relu1->output(0)));
    EXPECT_EQ(out.get_node(), f->get_results()[1].get());
    EXPECT_EQ(f->get_results().size(), 2);
    EXPECT_EQ(f->get_results()[1]->input_value(0).get_node(), relu1.get());
}

TEST(model, add_output_to_new_subgraph) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    relu1->set_friendly_name("relu1");
    relu1->get_output_tensor(0).set_names({"relu_t1"});

    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    relu2->set_friendly_name("relu2");
    relu2->get_output_tensor(0).set_names({"relu_t2"});
    auto f = std::make_shared<ov::Model>(relu2, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_EQ(f->get_results().size(), 1);

    ov::Output<ov::Node> out;
    EXPECT_NO_THROW(
        out = f->add_output(ov::opset8::Constant::create(ov::element::i32, {1}, std::vector<int32_t>{1})->output(0)));
    EXPECT_NO_THROW(f->get_ordered_ops());
    EXPECT_EQ(out.get_node(), f->get_results()[1].get());
    EXPECT_EQ(f->get_results().size(), 2);
}

TEST(model, add_output_incorrect_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    relu1->set_friendly_name("relu1");
    relu1->get_output_tensor(0).set_names({"relu_t1"});

    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    relu2->set_friendly_name("relu2");
    relu2->get_output_tensor(0).set_names({"relu_t2"});
    auto f = std::make_shared<ov::Model>(relu2, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_EQ(f->get_results().size(), 1);

    EXPECT_THROW(f->add_output("relu"), ov::Exception);
    EXPECT_EQ(f->get_results().size(), 1);
}

TEST(model, add_output_op_incorrect_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    relu1->set_friendly_name("relu1");
    relu1->get_output_tensor(0).set_names({"relu_t1"});

    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    relu2->set_friendly_name("relu2");
    relu2->get_output_tensor(0).set_names({"relu_t2"});
    auto f = std::make_shared<ov::Model>(relu2, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_EQ(f->get_results().size(), 1);

    EXPECT_THROW(f->add_output("relu_t1", 0), ov::Exception);
    EXPECT_EQ(f->get_results().size(), 1);
}

TEST(model, add_output_op_name_incorrect_idx) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    relu1->set_friendly_name("relu1");
    relu1->get_output_tensor(0).set_names({"relu_t1"});

    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    relu2->set_friendly_name("relu2");
    relu2->get_output_tensor(0).set_names({"relu_t2"});
    auto f = std::make_shared<ov::Model>(relu2, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_EQ(f->get_results().size(), 1);

    EXPECT_THROW(f->add_output("relu1", 10), ov::Exception);
    EXPECT_EQ(f->get_results().size(), 1);
}

TEST(model, add_output_port_to_result) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    relu1->set_friendly_name("relu1");
    relu1->get_output_tensor(0).set_names({"relu_t1"});

    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    relu2->set_friendly_name("relu2");
    relu2->get_output_tensor(0).set_names({"relu_t2"});
    auto result = std::make_shared<ov::opset8::Result>(relu2);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0});
    f->validate_nodes_and_infer_types();

    EXPECT_EQ(f->get_results().size(), 1);

    ov::Output<ov::Node> out;
    EXPECT_NO_THROW(out = f->add_output(result->output(0)));
    EXPECT_EQ(f->get_results().size(), 1);
    EXPECT_EQ(out, result->output(0));
}

TEST(model, add_output_performance) {
    // skip this test due to CVS-140440
    GTEST_SKIP() << "CVS-140440";
    using namespace std::chrono;
    auto test = [](int cnt, bool& timeout) -> size_t {
        auto shape = ov::Shape{1, 1, 224, 224};
        auto type = ov::element::f32;
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->set_friendly_name("Param1");
        param->output(0).set_names({"Param1"});
        ov::NodeVector nodes;
        std::shared_ptr<ov::Node> op = param;
        for (int i = 0; i < cnt; i++) {
            op = std::make_shared<ov::opset8::Add>(op, op);
            op->set_friendly_name("OpNameAdd" + std::to_string(i));
            op->output(0).set_names({"Add" + std::to_string(i)});
        }
        auto res = std::make_shared<ov::op::v0::Result>(op);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param});
        auto start = steady_clock::now();
        // Add outputs to all nodes via op+port and tensor names
        for (int i = 0; i < cnt; i++) {
            model->add_output("Add" + std::to_string(i));
            model->add_output("OpNameAdd" + std::to_string(i), 0);
            if (i % 100 == 0 && duration_cast<seconds>(steady_clock::now() - start).count() > 30) {
                timeout = true;
                return 0;
            }
        }
        auto end = steady_clock::now();
        return duration_cast<microseconds>(end - start).count();
    };
    bool timeout = false;
    auto t1 = test(200, timeout);
    EXPECT_FALSE(timeout);
    auto t2 = test(10000, timeout);  // Should be ~50 times longer, not 2500 times
    EXPECT_FALSE(timeout);
    EXPECT_LE(t2, t1 * 1000);  // Check 1000 times threshold (expected 50) which is definitely enough
}

TEST(model, add_output_cache_invalidation_tensor_name) {
    auto shape = ov::Shape{1, 1, 224, 224};
    auto type = ov::element::f32;
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    param->output(0).set_names({"Param1"});
    auto op = std::make_shared<ov::opset8::Add>(param, param);
    op->output(0).set_names({"TensorName"});
    auto op1 = std::make_shared<ov::opset8::Abs>(op);
    auto op2 = std::make_shared<ov::opset8::Relu>(op1);
    auto op3 = std::make_shared<ov::opset8::Abs>(op2);
    op3->output(0).set_names({"Tensor3"});
    auto res = std::make_shared<ov::op::v0::Result>(op3);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param});
    model->add_output("Tensor3");  // This creates cache

    op->output(0).set_names({"OldTensorName"});
    op2->output(0).set_names({"TensorName"});
    auto added = model->add_output("TensorName");  // This shall update cache as "TensorName" points to another node
    // Verify that output is added to 'op2'
    auto added_type_name = added.get_node_shared_ptr()->input(0).get_source_output().get_node()->get_type_name();
    EXPECT_EQ(ov::opset8::Relu::get_type_info_static().name, std::string(added_type_name));
}

TEST(model, add_output_cache_invalidation_op_name) {
    auto shape = ov::Shape{1, 1, 224, 224};
    auto type = ov::element::f32;
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    param->set_friendly_name("Param1");
    auto op = std::make_shared<ov::opset8::Add>(param, param);
    op->set_friendly_name("OpName");
    auto op1 = std::make_shared<ov::opset8::Abs>(op);
    auto op2 = std::make_shared<ov::opset8::Relu>(op1);
    auto op3 = std::make_shared<ov::opset8::Abs>(op2);
    op3->set_friendly_name("Op3");
    auto res = std::make_shared<ov::op::v0::Result>(op3);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param});
    model->add_output("Op3", 0);  // This creates cache

    op->set_friendly_name("OldOpName");
    op2->set_friendly_name("OpName");
    auto added = model->add_output("OpName", 0);  // This shall update cache as "OpName" points to another node
    // Verify that output is added to 'op2'
    auto added_type_name = added.get_node_shared_ptr()->input(0).get_source_output().get_node()->get_type_name();
    EXPECT_EQ(ov::opset8::Relu::get_type_info_static().name, std::string(added_type_name));
}

TEST(model, add_output_ordered_ops) {
    auto shape = ov::Shape{1, 1, 224, 224};
    auto type = ov::element::f32;
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    param->set_friendly_name("Param1");
    auto op = std::make_shared<ov::opset8::Add>(param, param);
    op->set_friendly_name("OpName");
    auto op1 = std::make_shared<ov::opset8::Abs>(op);
    auto op2 = std::make_shared<ov::opset8::Relu>(op1);
    auto op3 = std::make_shared<ov::opset8::Abs>(op2);
    op3->set_friendly_name("Op3");
    auto res = std::make_shared<ov::op::v0::Result>(op3);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param});
    auto ops_before = model->get_ordered_ops();
    auto new_res = model->add_output(op2);
    auto ops_after = model->get_ordered_ops();
    EXPECT_EQ(ops_after.size(), ops_before.size() + 1)
        << "Before: " << ops_before.size() << ". After: " << ops_after.size();
    bool relu_found = false, relu_result_found = false;
    for (const auto& node : ops_after) {
        if (ov::as_type_ptr<ov::opset8::Relu>(node)) {
            relu_found = true;
            EXPECT_FALSE(relu_result_found);
        } else if (ov::as_type_ptr<ov::opset8::Result>(node) &&
                   ov::as_type_ptr<ov::opset8::Relu>(node->get_input_node_shared_ptr(0))) {
            relu_result_found = true;
            EXPECT_TRUE(relu_found);
        }
    }
    EXPECT_TRUE(relu_found);
    EXPECT_TRUE(relu_result_found);
    // Invalidate result
    ops_before = model->get_ordered_ops();
    res->set_arguments(ov::NodeVector{});
    EXPECT_EQ(model->get_ordered_ops().size(), ops_before.size() - 1);
}

// Scenario:
// 1. Create model with nodes A,B,C.
// 2. Add output to some node 'A' - 'names cache' will be created
// 3. Remove some node 'B' - ordered_ops_cache will be 'false'
// 4. Assign name 'B' to existing node 'C'
// 5. Call get_ordered_ops (ordered_ops_cache=true)
// 6. Expect: Add_output to 'B' - output to node 'C' shall be added on step 4
// Without clearing 'names cache' on step 5 - test will incorrectly add output to 'B'
TEST(model, add_output_clear_cached_tensor_name_by_ordered_ops) {
    // 1. Create model with nodes A,B,C.
    auto shape = ov::Shape{1, 1, 224, 224};
    auto type = ov::element::f32;
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    auto op1 = std::make_shared<ov::opset8::Abs>(param);
    op1->get_default_output().set_names({"A"});
    auto op2 = std::make_shared<ov::opset8::Relu>(op1);
    op2->get_default_output().set_names({"B"});
    auto op3 = std::make_shared<ov::opset8::Abs>(op2);
    op3->get_default_output().set_names({"C"});
    auto op4 = std::make_shared<ov::opset8::Subtract>(op3, op3);
    op4->get_default_output().set_names({"D"});
    auto res = std::make_shared<ov::op::v0::Result>(op4);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param});
    // 2. Add output to some node 'A' - 'names cache' will be created
    auto a_output = model->add_output("A");
    auto ops_before = model->get_ordered_ops();
    // 3. Remove some node 'B' - ordered_ops_cache will be 'false'
    auto op2_new = std::make_shared<ov::opset8::Add>(op1, op1);
    model->replace_node(op2, op2_new);
    // 4. Assign name 'B' to existing node 'C'
    op3->get_default_output().set_names({"B"});
    // 5. Call get_ordered_ops (ordered_ops_cache=true)
    auto ops_after = model->get_ordered_ops();
    EXPECT_EQ(ops_after.size(), ops_before.size());
    // 6. Expect: Add_output to 'B' - output to node 'C' shall be added on step 4
    auto b_output = model->add_output("B");
    std::string b_type = b_output.get_node_shared_ptr()->get_input_node_shared_ptr(0)->get_type_name();
    EXPECT_EQ(b_type, op3->get_type_name());
}

// Same as above, but for add_output(opName, idx) case. Scenario:
// 1. Create model with nodes A,B,C.
// 2. Add output to some node 'A' - 'names cache' will be created
// 3. Remove some node 'B' - ordered_ops_cache will be 'false'
// 4. Assign name 'B' to existing node 'C'
// 5. Call get_ordered_ops (ordered_ops_cache=true)
// 6. Expect: Add_output to 'B' - output to node 'C' shall be added on step 4
// Without clearing 'names cache' on step 5 - test will incorrectly add output to 'B'
TEST(model, add_output_clear_cached_op_name_by_ordered_ops) {
    // 1. Create model with nodes A,B,C.
    auto shape = ov::Shape{1, 1, 224, 224};
    auto type = ov::element::f32;
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    auto op1 = std::make_shared<ov::opset8::Abs>(param);
    op1->set_friendly_name("A");
    auto op2 = std::make_shared<ov::opset8::Relu>(op1);
    op2->set_friendly_name("B");
    auto op3 = std::make_shared<ov::opset8::Abs>(op2);
    op3->set_friendly_name("C");
    auto op4 = std::make_shared<ov::opset8::Subtract>(op3, op3);
    op4->set_friendly_name("D");
    auto res = std::make_shared<ov::op::v0::Result>(op4);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param});
    // 2. Add output to some node 'A' - 'names cache' will be created
    auto a_output = model->add_output("A", 0);
    auto ops_before = model->get_ordered_ops();
    // 3. Remove some node 'B' - ordered_ops_cache will be 'false'
    auto op2_new = std::make_shared<ov::opset8::Add>(op1, op1);
    model->replace_node(op2, op2_new);
    // 4. Assign name 'B' to existing node 'C'
    op3->set_friendly_name("B");
    // 5. Call get_ordered_ops (ordered_ops_cache=true)
    auto ops_after = model->get_ordered_ops();
    EXPECT_EQ(ops_after.size(), ops_before.size());
    // 6. Expect: Add_output to 'B' - output to node 'C' shall be added on step 4
    auto b_output = model->add_output("B", 0);
    std::string b_type = b_output.get_node_shared_ptr()->get_input_node_shared_ptr(0)->get_type_name();
    EXPECT_EQ(b_type, op3->get_type_name());
}

namespace {
bool all_ops_have_same_info(const std::shared_ptr<ov::Model>& f) {
    auto shared_info = ov::ModelAccessor(f).get_shared_info();
    for (auto&& op : f->get_ordered_ops()) {
        if (std::set<std::shared_ptr<ov::SharedRTInfo>>({shared_info}) != ov::NodeAccessor(op).get_shared_info()) {
            return false;
        }
    }
    return true;
}
}  // namespace

TEST(model, topological_sort_throws_if_loop_with_one_node) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);

    // Loop relu1->relu1
    relu1->input(0).replace_source_output(relu1->output(0));

    auto result = std::make_shared<ov::opset8::Result>(relu1);
    ASSERT_THROW(std::ignore = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0}),
                 ov::Exception);
}

TEST(model, topological_sort_throws_if_loop_with_several_nodes) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    auto result = std::make_shared<ov::opset8::Result>(relu1);

    // Loop relu2->relu3->relu2
    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1->output(0));
    auto relu3 = std::make_shared<ov::opset8::Relu>(relu2);
    ov::replace_node(relu1, relu3);

    ASSERT_THROW(std::ignore = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0}),
                 ov::Exception);
}

TEST(model, topological_sort_throws_if_loop_with_control_dependency) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    auto result = std::make_shared<ov::opset8::Result>(relu2);

    // Loop relu1->relu2->relu1
    relu1->add_control_dependency(relu2);

    ASSERT_THROW(std::ignore = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0}),
                 ov::Exception);
}

TEST(model, topological_sort_throws_if_loop_with_control_dependency_only) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu0 = std::make_shared<ov::opset8::Relu>(arg0);
    auto result0 = std::make_shared<ov::opset8::Result>(relu0);

    auto arg1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg1);
    auto result1 = std::make_shared<ov::opset8::Result>(relu1);

    // Loop relu0->relu1->relu0
    relu0->add_control_dependency(relu1);
    relu1->add_control_dependency(relu0);

    ASSERT_THROW(
        std::ignore = std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, ov::ParameterVector{arg0, arg1}),
        ov::Exception);
}

TEST(model, topological_sort_caching_basic) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    auto result = std::make_shared<ov::opset8::Result>(relu2);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0});

    auto shared_info = ov::ModelAccessor(f).get_shared_info();
    // Check that after model creation which call get_ordered_ops
    // cache is set to true value
    ASSERT_TRUE(shared_info->get_use_topological_cache());

    // Check that nodes contains the same shared info after model creation
    ASSERT_EQ(ov::NodeAccessor(arg0).get_shared_info().size(), 1);
    ASSERT_TRUE(ov::NodeAccessor(arg0).get_shared_info().count(shared_info));

    ASSERT_EQ(ov::NodeAccessor(relu1).get_shared_info().size(), 1);
    ASSERT_TRUE(ov::NodeAccessor(relu1).get_shared_info().count(shared_info));

    ASSERT_EQ(ov::NodeAccessor(relu2).get_shared_info().size(), 1);
    ASSERT_TRUE(ov::NodeAccessor(relu2).get_shared_info().count(shared_info));

    ASSERT_EQ(ov::NodeAccessor(result).get_shared_info().size(), 1);
    ASSERT_TRUE(ov::NodeAccessor(result).get_shared_info().count(shared_info));

    ASSERT_EQ(f->get_ordered_ops().size(), 4);
}

TEST(model, topological_sort_caching_replace_node) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    auto result = std::make_shared<ov::opset8::Result>(relu2);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0});

    auto shared_info = ov::ModelAccessor(f).get_shared_info();
    ASSERT_TRUE(shared_info->get_use_topological_cache());

    auto new_relu = std::make_shared<ov::opset8::Relu>(relu1);
    ov::replace_node(relu2, new_relu);

    // model has changed so cache must be updated
    ASSERT_FALSE(shared_info->get_use_topological_cache());

    // Before get_ordered_ops, new_node shouldn't have shared_info, but after
    // it will be set to the model shared_info and cache will be used.
    ASSERT_FALSE(ov::NodeAccessor(new_relu).get_shared_info().count(shared_info));
    ASSERT_EQ(f->get_ordered_ops().size(), 4);
    ASSERT_TRUE(ov::NodeAccessor(new_relu).get_shared_info().count(shared_info));
    ASSERT_TRUE(shared_info->get_use_topological_cache());
    ASSERT_TRUE(all_ops_have_same_info(f));
}

TEST(model, topological_sort_caching_replace_source_output) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    auto result = std::make_shared<ov::opset8::Result>(relu2);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0});

    auto shared_info = ov::ModelAccessor(f).get_shared_info();
    ASSERT_TRUE(shared_info->get_use_topological_cache());

    relu2->input(0).replace_source_output(relu1);

    // model has changed so cache must be updated
    ASSERT_FALSE(shared_info->get_use_topological_cache());

    ASSERT_EQ(f->get_ordered_ops().size(), 4);
    ASSERT_TRUE(shared_info->get_use_topological_cache());
    ASSERT_TRUE(all_ops_have_same_info(f));
}

TEST(model, topological_sort_caching_dangling_node) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    auto result = std::make_shared<ov::opset8::Result>(relu2);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0});

    auto shared_info = ov::ModelAccessor(f).get_shared_info();
    ASSERT_TRUE(shared_info->get_use_topological_cache());

    auto new_relu = std::make_shared<ov::opset8::Relu>(relu1);

    // model has not changed so cache mustn't be updated
    ASSERT_TRUE(shared_info->get_use_topological_cache());
    // Dangling node is not in model
    ASSERT_EQ(f->get_ordered_ops().size(), 4);
}

TEST(model, topological_sort_caching_replace_output) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    auto result = std::make_shared<ov::opset8::Result>(relu2);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0});

    auto shared_info = ov::ModelAccessor(f).get_shared_info();
    ASSERT_TRUE(shared_info->get_use_topological_cache());

    auto new_relu = std::make_shared<ov::opset8::Relu>(relu1);
    relu2->output(0).replace(new_relu);

    // model has changed so cache must be updated
    ASSERT_FALSE(shared_info->get_use_topological_cache());
    ASSERT_EQ(f->get_ordered_ops().size(), 4);
    ASSERT_TRUE(shared_info->get_use_topological_cache());
    ASSERT_TRUE(all_ops_have_same_info(f));
}

TEST(model, topological_sort_caching_set_argument) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    auto result = std::make_shared<ov::opset8::Result>(relu2);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0});

    auto shared_info = ov::ModelAccessor(f).get_shared_info();
    ASSERT_TRUE(shared_info->get_use_topological_cache());

    relu2->set_argument(0, arg0);

    // model has changed so cache must be updated
    ASSERT_FALSE(shared_info->get_use_topological_cache());
    ASSERT_EQ(f->get_ordered_ops().size(), 3);
    ASSERT_TRUE(shared_info->get_use_topological_cache());
    ASSERT_TRUE(all_ops_have_same_info(f));
}

TEST(model, topological_sort_caching_set_arguments) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    auto result = std::make_shared<ov::opset8::Result>(relu2);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0});

    auto shared_info = ov::ModelAccessor(f).get_shared_info();
    ASSERT_TRUE(shared_info->get_use_topological_cache());

    relu2->set_arguments({arg0->output(0)});

    // model has changed so cache must be updated
    ASSERT_FALSE(shared_info->get_use_topological_cache());
    ASSERT_EQ(f->get_ordered_ops().size(), 3);
    ASSERT_TRUE(shared_info->get_use_topological_cache());
    ASSERT_TRUE(all_ops_have_same_info(f));
}

TEST(model, topological_sort_caching_add_cf) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    auto result = std::make_shared<ov::opset8::Result>(relu2);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0});

    auto shared_info = ov::ModelAccessor(f).get_shared_info();
    ASSERT_TRUE(shared_info->get_use_topological_cache());

    relu2->add_control_dependency(arg0);

    // model has changed so cache must be updated
    ASSERT_FALSE(shared_info->get_use_topological_cache());
    ASSERT_EQ(f->get_ordered_ops().size(), 4);
    ASSERT_TRUE(shared_info->get_use_topological_cache());
    ASSERT_TRUE(all_ops_have_same_info(f));
}

TEST(model, topological_sort_caching_result_parameter_sink) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg0);
    auto relu2 = std::make_shared<ov::opset8::Relu>(relu1);
    auto result = std::make_shared<ov::opset8::Result>(relu2);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{arg0});

    auto shared_info = ov::ModelAccessor(f).get_shared_info();
    ASSERT_TRUE(shared_info->get_use_topological_cache());

    auto check_caching_status = [=](int64_t expected_number_of_ops) {
        ASSERT_FALSE(shared_info->get_use_topological_cache());
        ASSERT_EQ(f->get_ordered_ops().size(), expected_number_of_ops);
        ASSERT_TRUE(shared_info->get_use_topological_cache());
        ASSERT_TRUE(all_ops_have_same_info(f));
    };

    auto result2 = std::make_shared<ov::opset8::Result>(relu2);
    f->add_results({result2});
    check_caching_status(5);

    f->remove_result(result2);
    check_caching_status(4);

    auto arg1 = std::make_shared<ov::opset8::Parameter>();
    f->add_parameters({arg1});
    check_caching_status(5);

    f->remove_parameter(arg1);
    check_caching_status(4);

    auto assign = std::make_shared<ov::opset8::Assign>();
    f->add_sinks({assign});
    check_caching_status(5);

    f->remove_sink(assign);
    check_caching_status(4);
}

TEST(model, topological_sort_caching_multiple_components) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu0 = std::make_shared<ov::opset8::Relu>(arg0);
    auto result0 = std::make_shared<ov::opset8::Result>(relu0);

    auto arg1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu1 = std::make_shared<ov::opset8::Relu>(arg1);
    auto result1 = std::make_shared<ov::opset8::Result>(relu1);

    auto f = std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, ov::ParameterVector{arg0, arg1});

    auto shared_info = ov::ModelAccessor(f).get_shared_info();
    ASSERT_TRUE(shared_info->get_use_topological_cache());
    ASSERT_TRUE(all_ops_have_same_info(f));
}

TEST(model, topological_sort_caching_shared_nodes) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    auto relu0 = std::make_shared<ov::opset8::Relu>(arg0);
    auto result0 = std::make_shared<ov::opset8::Result>(relu0);

    auto f1 = std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{arg0});
    auto f2 = std::make_shared<ov::Model>(ov::ResultVector{result0}, ov::ParameterVector{arg0});

    auto f1_shared_info = ov::ModelAccessor(f1).get_shared_info();
    auto f2_shared_info = ov::ModelAccessor(f2).get_shared_info();

    for (auto&& node : f1->get_ordered_ops()) {
        auto node_info = ov::NodeAccessor(node).get_shared_info();
        // As two models owns the same node so node will have two shared_info objects
        ASSERT_EQ(node_info.size(), 2);
        ASSERT_TRUE(node_info.count(f1_shared_info));
        ASSERT_TRUE(node_info.count(f2_shared_info));
    }

    relu0->add_control_dependency(arg0);  // simply drop cache
    ASSERT_FALSE(f1_shared_info->get_use_topological_cache());
    ASSERT_FALSE(f2_shared_info->get_use_topological_cache());
}

namespace bs_utils {
static std::shared_ptr<ov::Model> create_n_inputs(ov::element::Type type,
                                                  const std::vector<ov::PartialShape>& shapes,
                                                  const std::vector<ov::Layout>& layouts) {
    ov::ResultVector res;
    ov::ParameterVector params;
    for (size_t i = 0; i < shapes.size(); i++) {
        auto index_str = std::to_string(i);
        auto data1 = std::make_shared<ov::opset8::Parameter>(type, shapes[i]);
        data1->set_layout(layouts[i]);
        data1->set_friendly_name("input" + index_str);
        data1->get_output_tensor(0).set_names({"tensor_input" + index_str});
        auto op1 = std::make_shared<ov::opset8::Relu>(data1);
        op1->set_friendly_name("Relu" + index_str);
        auto res1 = std::make_shared<ov::opset8::Result>(op1);
        res1->set_friendly_name("Result" + index_str);
        res1->get_output_tensor(0).set_names({"tensor_output" + index_str});
        params.push_back(data1);
        res.push_back(res1);
    }
    auto f = std::make_shared<ov::Model>(res, params);
    f->validate_nodes_and_infer_types();
    return f;
}

static std::shared_ptr<ov::Model> create_add(ov::element::Type type,
                                             const ov::PartialShape& shape,
                                             const ov::Layout& layout1,
                                             const ov::Layout& layout2) {
    ov::ParameterVector params;
    for (size_t i = 0; i < 2; i++) {
        auto index_str = std::to_string(i);
        auto data1 = std::make_shared<ov::opset8::Parameter>(type, shape);
        data1->set_friendly_name("input" + index_str);
        data1->get_output_tensor(0).set_names({"tensor_input" + index_str});
        params.push_back(data1);
    }
    params[0]->set_layout(layout1);
    params[1]->set_layout(layout2);
    auto op1 = std::make_shared<ov::opset8::Add>(params[0],
                                                 params[1],
                                                 ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::EXPLICIT));
    op1->set_friendly_name("Add");
    auto res1 = std::make_shared<ov::opset8::Result>(op1);
    res1->get_output_tensor(0).set_names({"tensor_output"});
    auto f = std::make_shared<ov::Model>(res1, params);
    f->validate_nodes_and_infer_types();
    return f;
}
}  // namespace bs_utils

TEST(model, get_batch_size) {
    auto f = bs_utils::create_n_inputs(ov::element::f32, {{1, 512, 512, 3}, {1, 3, 224, 224}}, {"NHWC", "NCHW"});

    EXPECT_NO_THROW(ov::get_batch(f));
    EXPECT_EQ(ov::get_batch(f), 1);
}

TEST(model, get_batch_size_with_conflict) {
    auto f = bs_utils::create_n_inputs(ov::element::f32,
                                       {ov::PartialShape::dynamic(), {5, 6}, {1, 3, 224, 224}, {3, 1}},
                                       {"NCHW", "D...", "NCHW", "N?"});

    // TODO: gtest v.10 limitation. Replace with EXPECT_THAT for gtest >= v1.11
    try {
        ov::get_batch(f);
        FAIL() << "get_batch shall throw";
    } catch (const ov::Exception& err) {
        // Verify error message contains conflicting layouts
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout("NCHW").to_string()) != std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout("N?").to_string()) != std::string::npos) << err.what();
        // Verify error message doesn't contain non-conflicting layouts
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout("D...").to_string()) == std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find("tensor_input_0") == std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find("tensor_input_1") == std::string::npos) << err.what();
    } catch (...) {
        FAIL() << "Expected ov::Exception";
    }
}

TEST(model, get_batch_size_without_batches) {
    auto f = bs_utils::create_n_inputs(ov::element::f32, {{1, 3, 224, 224}, {1, 3, 224, 224}}, {"?C...", ov::Layout()});

    // TODO: replace with EXPECT_THAT after upgrade gtest to v1.11
    try {
        ov::get_batch(f);
        FAIL() << "get_batch shall throw";
    } catch (const ov::Exception& err) {
        // Verify error message contains layouts without batches
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout("?C...").to_string()) != std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout().to_string()) != std::string::npos) << err.what();
    } catch (...) {
        FAIL() << "Expected ov::Exception";
    }
}

TEST(model, get_batch_size_without_one_layout) {
    auto f = bs_utils::create_n_inputs(ov::element::f32,
                                       {{ov::Dimension::dynamic(), 3, 224, 224}, {10, 20}},
                                       {"N...", "HW"});
    EXPECT_EQ(ov::get_batch(f), ov::Dimension::dynamic());
}

TEST(model, get_batch_size_ranges) {
    auto f = bs_utils::create_n_inputs(ov::element::f32,
                                       {{ov::Dimension(1, 10), 3, 224, 224}, {ov::Dimension(5, 15), 3, 224, 224}},
                                       {"NCHW", "NCHW"});
    EXPECT_EQ(ov::get_batch(f), ov::Dimension(5, 10));
}

TEST(model, set_batch_size) {
    auto f = bs_utils::create_n_inputs(ov::element::f32,
                                       {{1, 512, 512, 3}, {ov::Dimension::dynamic(), 3, 224, 224}, {1, 5}},
                                       {"NHWC", "NCHW", "??"});
    EXPECT_NO_THROW(ov::set_batch(f, 4));
    ov::PartialShape pshape({1, 4, 3, 3});
    EXPECT_EQ(f->input("tensor_input0").get_partial_shape(), (ov::PartialShape{4, 512, 512, 3}));
    EXPECT_EQ(f->input("tensor_input1").get_partial_shape(), (ov::PartialShape{4, 3, 224, 224}));
    EXPECT_EQ(f->input("tensor_input2").get_partial_shape(), (ov::PartialShape{1, 5}));
}

TEST(model, set_batch_size_ranges) {
    auto f = bs_utils::create_n_inputs(ov::element::f32,
                                       {{ov::Dimension(1, 10), 3, 224, 224}, {ov::Dimension(5, 15), 3, 224, 224}},
                                       {"NCHW", "NCHW"});
    EXPECT_NO_THROW(ov::set_batch(f, 42));
    EXPECT_EQ(f->input("tensor_input0").get_partial_shape(), (ov::PartialShape{42, 3, 224, 224}));
    EXPECT_EQ(f->input("tensor_input1").get_partial_shape(), (ov::PartialShape{42, 3, 224, 224}));
}

TEST(model, set_batch_size_fully_dynamic) {
    auto f =
        bs_utils::create_n_inputs(ov::element::f32, {ov::PartialShape::dynamic(), {1, 3, 224, 224}}, {"NCHW", "NCHW"});
    EXPECT_NO_THROW(ov::set_batch(f, 42));
    EXPECT_EQ(f->input("tensor_input0").get_partial_shape(), (ov::PartialShape::dynamic()));
    EXPECT_EQ(f->input("tensor_input1").get_partial_shape(), (ov::PartialShape{42, 3, 224, 224}));
}

TEST(model, set_batch_size_dynamic_layout) {
    auto f = bs_utils::create_n_inputs(ov::element::f32, {{3, 224, 224, 1}, {1, 3, 224, 224}}, {"...N", "NCHW"});
    EXPECT_NO_THROW(ov::set_batch(f, 42));
    EXPECT_EQ(f->input("tensor_input0").get_partial_shape(), (ov::PartialShape{3, 224, 224, 42}));
    EXPECT_EQ(f->input("tensor_input1").get_partial_shape(), (ov::PartialShape{42, 3, 224, 224}));
}

TEST(model, set_batch_size_with_conflict) {
    auto f = bs_utils::create_n_inputs(ov::element::f32,
                                       {ov::PartialShape::dynamic(), {5, 6}, {1, 3, 224, 224}, {3, 1}},
                                       {"NCHW", "D...", "NCHW", "N?"});

    // TODO: gtest v.10 limitation. Replace with EXPECT_THAT for gtest >= v1.11
    try {
        ov::set_batch(f, 12);
        FAIL() << "set_batch shall throw";
    } catch (const ov::Exception& err) {
        // Verify error message contains conflicting layouts
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout("NCHW").to_string()) != std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout("N?").to_string()) != std::string::npos) << err.what();
        // Verify error message doesn't contain non-conflicting layouts
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout("D...").to_string()) == std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find("tensor_input_0") == std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find("tensor_input_1") == std::string::npos) << err.what();
    } catch (...) {
        FAIL() << "Expected ov::Exception";
    }
}

TEST(model, set_batch_size_without_batches) {
    auto f = bs_utils::create_n_inputs(ov::element::f32, {{1, 3, 224, 224}, {1, 3, 224, 224}}, {"?C...", ov::Layout()});

    // TODO: replace with EXPECT_THAT after upgrade gtest to v1.11
    try {
        ov::set_batch(f, 42);
        FAIL() << "set_batch shall throw";
    } catch (const ov::Exception& err) {
        // Verify error message contains layouts without batches
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout("?C...").to_string()) != std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout().to_string()) != std::string::npos) << err.what();
    } catch (...) {
        FAIL() << "Expected ov::Exception";
    }
}

TEST(model, set_batch_size_validation_throw) {
    auto f = bs_utils::create_add(ov::element::f32, {1, 3, 224, 224}, "NCHW", ov::Layout());

    // TODO: replace with EXPECT_THAT after upgrade gtest to v1.11
    try {
        ov::set_batch(f, 42);
        FAIL() << "set_batch shall throw";
    } catch (const ov::Exception& err) {
        // Verify error message contains possible reasons
        EXPECT_TRUE(std::string(err.what()).find("Possible reasons") != std::string::npos) << err.what();
        // Verify error message contains all layouts
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout("NCHW").to_string()) != std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find(ov::Layout().to_string()) != std::string::npos) << err.what();
    } catch (...) {
        FAIL() << "Expected ov::Exception";
    }
}

TEST(model, incompatible_layout) {
    auto f = bs_utils::create_n_inputs(ov::element::f32, {{1, 3, 224, 224}}, {"NCHW"});
    using callback = std::function<void()>;
    auto verify_ex = [&](const callback& cb, const std::string& msg) {
        try {
            cb();
            FAIL() << "set_layout shall throw";
        } catch (const ov::Exception& err) {
            // Verify error message contains conflicting layouts
            EXPECT_TRUE(std::string(err.what()).find(msg) != std::string::npos) << err.what();
        } catch (...) {
            FAIL() << "Expected ov::Exception";
        }
    };
    auto verify_ex_set_layout = [&](const ov::Layout& layout) {
        auto msg = layout.to_string();
        verify_ex(
            [&]() {
                ov::layout::set_layout(f->input(), layout);
            },
            msg);
    };
    verify_ex_set_layout("HWC");
    verify_ex_set_layout("NDCHW");
    verify_ex_set_layout("ND...CHW");
    EXPECT_NO_THROW(ov::layout::set_layout(f->input(), "H...WC"));
    EXPECT_NO_THROW(ov::layout::set_layout(f->input(), "...NCHW"));
    EXPECT_NO_THROW(f->get_parameters()[0]->set_layout("NCHW..."));
    EXPECT_NO_THROW(f->get_parameters()[0]->set_layout("NCHW"));

    auto verify_ex_set_layout_param = [&](const ov::Layout& layout) {
        auto msg = layout.to_string();
        verify_ex(
            [&]() {
                f->get_parameters()[0]->set_layout(layout);
            },
            msg);
    };
    verify_ex_set_layout_param("HWC");
    verify_ex_set_layout_param("NDCHW");
    verify_ex_set_layout_param("ND...CHW");

    auto verify_ex_set_partial_shape = [&](const ov::PartialShape& shape) {
        std::stringstream msgStr;
        msgStr << shape;
        auto msg = msgStr.str();
        verify_ex(
            [&]() {
                f->get_parameters()[0]->set_partial_shape(shape);
            },
            msg);
    };
    verify_ex_set_partial_shape({1, 2, 3, 4, 5});
    verify_ex_set_partial_shape({1, 2, 3});
    EXPECT_NO_THROW(f->get_parameters()[0]->set_partial_shape(ov::PartialShape::dynamic()));
    EXPECT_NO_THROW(f->get_parameters()[0]->set_partial_shape(ov::PartialShape{1, 3, 224, 224}));

    auto verify_ex_set_layout_result = [&](const ov::Layout& layout) {
        auto msg = layout.to_string();
        verify_ex(
            [&]() {
                ov::layout::set_layout(f->output(), layout);
            },
            msg);
    };
    verify_ex_set_layout_result("HWC");
    verify_ex_set_layout_result("NDCHW");
    verify_ex_set_layout_result("ND...CHW");

    auto verify_ex_set_layout_result_validate = [&](const ov::PartialShape& param_shape, const ov::Layout& layout) {
        auto msg = layout.to_string();
        f = bs_utils::create_n_inputs(ov::element::f32, {ov::PartialShape::dynamic()}, {"..."});
        verify_ex(
            [&]() {
                f->get_parameters()[0]->set_partial_shape(param_shape);
                ov::layout::set_layout(f->output(), layout);
                f->validate_nodes_and_infer_types();
            },
            msg);
    };
    verify_ex_set_layout_result_validate({1, 2, 3, 4}, "HWC");
    verify_ex_set_layout_result_validate({1, 2, 3, 4}, "NDHWC");
    verify_ex_set_layout_result_validate({1, 2, 3, 4}, "ND...HWC");
}

TEST(model, clone_model_function) {
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
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    model->validate_nodes_and_infer_types();

    auto input1 = model->input(0);
    auto input2 = model->input("data1");

    auto cloned_model = model->clone();

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, cloned_model);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(model, clone_model) {
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
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{arg0, arg1});

    model->validate_nodes_and_infer_types();

    auto input1 = model->input(0);
    auto input2 = model->input("data1");

    auto cloned_model = model->clone();

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, cloned_model);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(model, set_meta_information) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Model>(relu, ov::ParameterVector{arg0});

    std::string key = "data";
    EXPECT_FALSE(f->has_rt_info(key, "test"));
    EXPECT_THROW(f->get_rt_info<std::string>(key, "test"), ov::Exception);

    EXPECT_FALSE(f->has_rt_info(key, "test1"));
    EXPECT_THROW(f->get_rt_info<std::string>(key, "test1"), ov::Exception);

    EXPECT_FALSE(f->has_rt_info({key, "test1"}));
    EXPECT_THROW(f->get_rt_info<std::string>({key, "test1"}), ov::Exception);

    f->set_rt_info("test_value", key, "test");
    f->set_rt_info("1", {key, "test1"});

    EXPECT_TRUE(f->has_rt_info(key, "test"));
    EXPECT_NO_THROW(f->get_rt_info<std::string>(key, "test"));
    EXPECT_EQ(f->get_rt_info<std::string>(key, "test"), "test_value");
    EXPECT_THROW(f->get_rt_info<int>(key, "test"), ov::Exception);

    EXPECT_TRUE(f->has_rt_info(key, "test1"));
    EXPECT_NO_THROW(f->get_rt_info<std::string>(key, "test1"));
    EXPECT_EQ(f->get_rt_info<std::string>(key, "test1"), "1");
    EXPECT_EQ(f->get_rt_info<int>(key, "test1"), 1);

    EXPECT_TRUE(f->has_rt_info({key, "test1"}));
    EXPECT_NO_THROW(f->get_rt_info<std::string>({key, "test1"}));
    EXPECT_EQ(f->get_rt_info<std::string>({key, "test1"}), "1");
    EXPECT_EQ(f->get_rt_info<int>({key, "test1"}), 1);
}

TEST(model, set_complex_meta_information) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Model>(relu, ov::ParameterVector{arg0});

    const auto check_rt_info = [](const std::shared_ptr<ov::Model>& model) {
        EXPECT_TRUE(model->has_rt_info("config", "type_of_model"));
        EXPECT_TRUE(model->has_rt_info("config", "converter_type"));
        EXPECT_TRUE(model->has_rt_info("config", "model_parameters", "threshold"));
        EXPECT_TRUE(model->has_rt_info("config", "model_parameters", "min"));
        EXPECT_TRUE(model->has_rt_info("config", "model_parameters", "max"));
        EXPECT_TRUE(model->has_rt_info("config", "model_parameters", "labels", "label_tree", "type"));
        EXPECT_TRUE(model->has_rt_info("config", "model_parameters", "labels", "label_tree", "directed"));
        EXPECT_TRUE(model->has_rt_info("config", "model_parameters", "labels", "label_tree", "nodes"));
        EXPECT_TRUE(model->has_rt_info("config", "model_parameters", "labels", "label_groups", "ids"));
        EXPECT_TRUE(model->has_rt_info("config", "model_parameters", "mean_values"));

        EXPECT_EQ("classification", model->get_rt_info<std::string>("config", "type_of_model"));
        EXPECT_EQ("classification", model->get_rt_info<std::string>("config", "converter_type"));
        EXPECT_FLOAT_EQ(13.23f, model->get_rt_info<float>("config", "model_parameters", "threshold"));
        EXPECT_FLOAT_EQ(-3.245433f, model->get_rt_info<float>("config", "model_parameters", "min"));
        EXPECT_FLOAT_EQ(3.2342233f, model->get_rt_info<float>("config", "model_parameters", "max"));
        EXPECT_EQ("tree",
                  model->get_rt_info<std::string>("config", "model_parameters", "labels", "label_tree", "type"));
        EXPECT_EQ(true, model->get_rt_info<bool>("config", "model_parameters", "labels", "label_tree", "directed"));
        EXPECT_EQ(std::vector<std::string>{},
                  model->get_rt_info<std::vector<std::string>>("config",
                                                               "model_parameters",
                                                               "labels",
                                                               "label_tree",
                                                               "nodes"));
        std::vector<std::string> str_vec{"sasd", "fdfdfsdf"};
        EXPECT_EQ(str_vec,
                  model->get_rt_info<std::vector<std::string>>("config",
                                                               "model_parameters",
                                                               "labels",
                                                               "label_groups",
                                                               "ids"));
        std::vector<float> fl_vec{22.3f, 33.11f, 44.f};
        EXPECT_EQ(fl_vec, model->get_rt_info<std::vector<float>>("config", "model_parameters", "mean_values"));
    };

    // Fill meta data
    f->set_rt_info("classification", "config", "type_of_model");
    f->set_rt_info("classification", "config", "converter_type");
    f->set_rt_info(13.23f, "config", "model_parameters", "threshold");
    f->set_rt_info(-3.245433f, "config", "model_parameters", "min");
    f->set_rt_info(3.2342233f, "config", "model_parameters", "max");
    f->set_rt_info("tree", "config", "model_parameters", "labels", "label_tree", "type");
    f->set_rt_info(true, "config", "model_parameters", "labels", "label_tree", "directed");
    f->set_rt_info(std::vector<std::string>{}, "config", "model_parameters", "labels", "label_tree", "nodes");
    f->set_rt_info(std::vector<std::string>{"sasd", "fdfdfsdf"},
                   "config",
                   "model_parameters",
                   "labels",
                   "label_groups",
                   "ids");
    f->set_rt_info(std::vector<float>{22.3f, 33.11f, 44.f}, "config", "model_parameters", "mean_values");

    check_rt_info(f);
}

TEST(model, create_model) {
    EXPECT_NO_THROW(ov::Model({}, ""));
    EXPECT_THROW(ov::Model(ov::ResultVector{nullptr}, {}, ""), ov::Exception);
    EXPECT_THROW(ov::Model(nullptr, {}, ""), ov::Exception);
    EXPECT_NO_THROW(ov::Model(ov::ResultVector{}, ov::ParameterVector{}, ""));
    EXPECT_THROW(ov::Model({nullptr}, {nullptr}, {nullptr}, {nullptr}, ""), ov::Exception);
    EXPECT_THROW(ov::Model({nullptr}, {}, {}, {}, ""), ov::Exception);
    EXPECT_THROW(ov::Model(ov::ResultVector{}, {nullptr}, {}, {}, ""), ov::Exception);
    EXPECT_THROW(ov::Model(ov::ResultVector{}, {}, {nullptr}, {}, ""), ov::Exception);
    EXPECT_THROW(ov::Model(ov::ResultVector{}, {}, {}, {nullptr}, ""), ov::Exception);
    EXPECT_THROW(ov::Model(ov::OutputVector{ov::Output<ov::Node>{nullptr, 0}}, {}, {}, {}, ""), ov::Exception);
}

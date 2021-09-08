// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/function.hpp"

#include <gtest/gtest.h>

#include "openvino/opsets/opset8.hpp"

TEST(function, get_input_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});

    auto input = f->input("input");
    ASSERT_EQ(input.get_node(), arg0.get());
    ASSERT_EQ(f->input_element_type("input"), ov::element::f32);
    ASSERT_EQ(f->input_shape("input"), ov::Shape{1});
}

TEST(function, get_output_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<ov::Function>(result, ov::ParameterVector{arg0});

    auto output = f->output("relu_t");
    ASSERT_EQ(output.get_node(), result.get());
    ASSERT_EQ(f->output_element_type("identity"), ov::element::f32);
    ASSERT_EQ(f->output_shape("identity"), ov::Shape{1});
}

TEST(function, get_incorrect_output_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});

    ASSERT_THROW(f->output("input"), ov::Exception);
}

TEST(function, get_incorrect_input_by_tensor_name) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});

    ASSERT_THROW(f->input("relu_t"), ov::Exception);
}

TEST(function, get_input_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});

    auto input = f->input(0);
    ASSERT_EQ(input.get_node(), arg0.get());
    ASSERT_EQ(f->input_element_type(0), ov::element::f32);
    ASSERT_EQ(f->input_shape(0), ov::Shape{1});
}

TEST(function, get_output_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto result = std::make_shared<ov::opset8::Result>(relu);
    auto f = std::make_shared<ov::Function>(result, ov::ParameterVector{arg0});

    auto output = f->output(0);
    ASSERT_EQ(output.get_node(), result.get());
    ASSERT_EQ(f->output_element_type(0), ov::element::f32);
    ASSERT_EQ(f->output_shape(0), ov::Shape{1});
}

TEST(function, get_incorrect_output_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});

    ASSERT_THROW(f->output(2), std::exception);
}

TEST(function, get_incorrect_input_by_index) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<ov::opset8::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f = std::make_shared<ov::Function>(relu, ov::ParameterVector{arg0});

    ASSERT_THROW(f->input(2), std::exception);
}

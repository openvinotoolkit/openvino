// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_bitwise_to_logical_bool.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

namespace {

std::shared_ptr<ov::Model> create_bitwise_model(std::string op_type, const ov::element::Type input_type) {
    const auto lhs = std::make_shared<ov::opset13::Parameter>(input_type, ov::Shape{1, 3, 100, 100});
    const auto rhs = std::make_shared<ov::opset13::Parameter>(input_type, ov::Shape{1, 3, 100, 100});

    std::shared_ptr<ov::Node> bitwise;
    ParameterVector params{lhs, rhs};
    if (op_type == "and") {
        bitwise = std::make_shared<ov::opset13::BitwiseAnd>(lhs, rhs, op::AutoBroadcastType::NONE);
    } else if (op_type == "not") {
        bitwise = std::make_shared<ov::opset13::BitwiseNot>(lhs);
        params = {lhs};
    } else if (op_type == "or") {
        bitwise = std::make_shared<ov::opset13::BitwiseOr>(lhs, rhs, op::AutoBroadcastType::NONE);
    } else if (op_type == "xor") {
        bitwise = std::make_shared<ov::opset13::BitwiseXor>(lhs, rhs, op::AutoBroadcastType::NONE);
    }

    bitwise->set_friendly_name("bitwise");

    return std::make_shared<ov::Model>(bitwise->outputs(), params);
}

std::shared_ptr<ov::Model> create_logical_model(std::string op_type) {
    const auto lhs = std::make_shared<ov::opset1::Parameter>(ov::element::boolean, ov::Shape{1, 3, 100, 100});
    const auto rhs = std::make_shared<ov::opset1::Parameter>(ov::element::boolean, ov::Shape{1, 3, 100, 100});
    std::shared_ptr<ov::Node> logical;
    ParameterVector params = {lhs, rhs};
    if (op_type == "and") {
        logical = std::make_shared<ov::opset1::LogicalAnd>(lhs, rhs, op::AutoBroadcastType::NONE);
    } else if (op_type == "not") {
        logical = std::make_shared<ov::opset1::LogicalNot>(lhs);
        params = {lhs};
    } else if (op_type == "or") {
        logical = std::make_shared<ov::opset1::LogicalOr>(lhs, rhs, op::AutoBroadcastType::NONE);
    } else if (op_type == "xor") {
        logical = std::make_shared<ov::opset1::LogicalXor>(lhs, rhs, op::AutoBroadcastType::NONE);
    }

    logical->set_friendly_name("logical");

    return std::make_shared<ov::Model>(logical->outputs(), params);
}

}  // namespace

TEST_F(TransformationTestsF, ConvertBitwiseToLogical_and_i32) {
    auto transform = manager.register_pass<ov::pass::GraphRewrite>();
    transform->add_matcher<ConvertBitwiseToLogical>();
    model = create_bitwise_model("and", element::i32);
}

TEST_F(TransformationTestsF, ConvertBitwiseToLogical_not_i32) {
    auto transform = manager.register_pass<ov::pass::GraphRewrite>();
    transform->add_matcher<ConvertBitwiseToLogical>();
    model = create_bitwise_model("not", element::i32);
}

TEST_F(TransformationTestsF, ConvertBitwiseToLogical_or_i32) {
    auto transform = manager.register_pass<ov::pass::GraphRewrite>();
    transform->add_matcher<ConvertBitwiseToLogical>();
    model = create_bitwise_model("or", element::i32);
}

TEST_F(TransformationTestsF, ConvertBitwiseToLogical_xor_i32) {
    auto transform = manager.register_pass<ov::pass::GraphRewrite>();
    transform->add_matcher<ConvertBitwiseToLogical>();
    model = create_bitwise_model("xor", element::i32);
}

TEST_F(TransformationTestsF, ConvertBitwiseToLogical_and_boolean) {
    auto transform = manager.register_pass<ov::pass::GraphRewrite>();
    transform->add_matcher<ConvertBitwiseToLogical>();
    model = create_bitwise_model("and", element::boolean);
    model_ref = create_logical_model("and");
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertBitwiseToLogical_not_boolean) {
    auto transform = manager.register_pass<ov::pass::GraphRewrite>();
    transform->add_matcher<ConvertBitwiseToLogical>();
    model = create_bitwise_model("not", element::boolean);
    model_ref = create_logical_model("not");
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertBitwiseToLogical_or_boolean) {
    auto transform = manager.register_pass<ov::pass::GraphRewrite>();
    transform->add_matcher<ConvertBitwiseToLogical>();
    model = create_bitwise_model("or", element::boolean);
    model_ref = create_logical_model("or");
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertBitwiseToLogical_xor_boolean) {
    auto transform = manager.register_pass<ov::pass::GraphRewrite>();
    transform->add_matcher<ConvertBitwiseToLogical>();
    model = create_bitwise_model("xor", element::boolean);
    model_ref = create_logical_model("xor");
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

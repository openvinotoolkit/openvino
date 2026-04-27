// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/utils/extract_subgraph.hpp"

#include <gtest/gtest.h>

#include <map>
#include <memory>

#include "common_test_utils/graph_comparator.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"

namespace op_util = ov::op::util;

// Builds:  param_a -> relu -> add -> result
//                    param_b -^
static std::shared_ptr<ov::Model> build_linear_model() {
    auto param_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    param_a->set_friendly_name("A");

    auto relu = std::make_shared<ov::op::v0::Relu>(param_a);
    relu->set_friendly_name("Relu");

    auto param_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    param_b->set_friendly_name("B");

    auto add = std::make_shared<ov::op::v1::Add>(relu, param_b);
    add->set_friendly_name("Add");

    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("Result");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_a, param_b});
}

// Builds a branching model:
//   param_a -> relu ---------> mul -> result
//   param_b -> sigmoid -------^
static std::shared_ptr<ov::Model> build_branch_model() {
    auto param_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    param_a->set_friendly_name("A");

    auto relu = std::make_shared<ov::op::v0::Relu>(param_a);
    relu->set_friendly_name("Relu");

    auto param_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    param_b->set_friendly_name("B");

    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(param_b);
    sigmoid->set_friendly_name("Sigmoid");

    auto mul = std::make_shared<ov::op::v1::Multiply>(relu, sigmoid);
    mul->set_friendly_name("Mul");

    auto result = std::make_shared<ov::op::v0::Result>(mul);
    result->set_friendly_name("Result");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_a, param_b});
}

TEST(ExtractSubgraphTest, CoreOverload_SingleOp) {
    auto model = build_linear_model();

    ov::op::v0::Relu* relu_ptr = nullptr;
    for (const auto& op : model->get_ordered_ops()) {
        if (op->get_friendly_name() == "Relu")
            relu_ptr = ov::as_type<ov::op::v0::Relu>(op.get());
    }
    ASSERT_NE(relu_ptr, nullptr);

    auto subgraph = op_util::extract_subgraph(model, {relu_ptr->input(0)}, {relu_ptr->output(0)});

    // Build the expected model: param -> relu -> result
    auto expected_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4});
    auto expected_relu = std::make_shared<ov::op::v0::Relu>(expected_param);
    expected_relu->set_friendly_name("Relu");
    auto expected_result = std::make_shared<ov::op::v0::Result>(expected_relu);
    auto expected = std::make_shared<ov::Model>(ov::ResultVector{expected_result}, ov::ParameterVector{expected_param});

    const auto cmp = FunctionsComparator::all_flags_enabled();
    const auto res = cmp.compare(subgraph, expected);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(ExtractSubgraphTest, CoreOverload_TwoInputsOneOutput) {
    auto model = build_linear_model();

    ov::op::v1::Add* add_ptr = nullptr;
    for (const auto& op : model->get_ordered_ops()) {
        if (op->get_friendly_name() == "Add")
            add_ptr = ov::as_type<ov::op::v1::Add>(op.get());
    }
    ASSERT_NE(add_ptr, nullptr);

    auto subgraph = op_util::extract_subgraph(model, {add_ptr->input(0), add_ptr->input(1)}, {add_ptr->output(0)});

    auto p0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4});
    auto p1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4});
    auto expected_add = std::make_shared<ov::op::v1::Add>(p0, p1);
    expected_add->set_friendly_name("Add");
    auto expected_result = std::make_shared<ov::op::v0::Result>(expected_add);
    auto expected = std::make_shared<ov::Model>(ov::ResultVector{expected_result}, ov::ParameterVector{p0, p1});

    const auto cmp = FunctionsComparator::all_flags_enabled().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = cmp.compare(subgraph, expected);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(ExtractSubgraphTest, CoreOverload_OriginalModelUnchanged) {
    auto model = build_linear_model();
    const size_t original_op_count = model->get_ordered_ops().size();

    ov::op::v0::Relu* relu_ptr = nullptr;
    for (const auto& op : model->get_ordered_ops()) {
        if (op->get_friendly_name() == "Relu")
            relu_ptr = ov::as_type<ov::op::v0::Relu>(op.get());
    }
    ASSERT_NE(relu_ptr, nullptr);

    op_util::extract_subgraph(model, {relu_ptr->input(0)}, {relu_ptr->output(0)});

    EXPECT_EQ(model->get_ordered_ops().size(), original_op_count);
    EXPECT_EQ(model->get_parameters().size(), 2u);
    EXPECT_EQ(model->get_results().size(), 1u);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(relu_ptr->get_input_node_shared_ptr(0)));
}

TEST(ExtractSubgraphTest, MultimapOverload_SingleOp) {
    auto model = build_linear_model();

    const std::multimap<std::string, size_t> inputs_map = {{"Relu", 0}};
    const std::multimap<std::string, size_t> outputs_map = {{"Relu", 0}};

    auto subgraph = op_util::extract_subgraph(model, inputs_map, outputs_map);

    auto expected_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4});
    auto expected_relu = std::make_shared<ov::op::v0::Relu>(expected_param);
    expected_relu->set_friendly_name("Relu");
    auto expected_result = std::make_shared<ov::op::v0::Result>(expected_relu);
    auto expected = std::make_shared<ov::Model>(ov::ResultVector{expected_result}, ov::ParameterVector{expected_param});

    const auto cmp = FunctionsComparator::all_flags_enabled();
    const auto res = cmp.compare(subgraph, expected);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(ExtractSubgraphTest, MultimapOverload_MultiInputChain) {
    auto model = build_linear_model();

    // Extract the whole Relu->Add chain: cut before Relu and before Add.input(1), output at Add
    const std::multimap<std::string, size_t> inputs_map = {{"Relu", 0}, {"Add", 1}};
    const std::multimap<std::string, size_t> outputs_map = {{"Add", 0}};

    auto subgraph = op_util::extract_subgraph(model, inputs_map, outputs_map);

    // Expected: p0 -> relu -> add(p1) -> result
    auto p0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4});
    auto p1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4});
    auto expected_relu = std::make_shared<ov::op::v0::Relu>(p0);
    expected_relu->set_friendly_name("Relu");
    auto expected_add = std::make_shared<ov::op::v1::Add>(expected_relu, p1);
    expected_add->set_friendly_name("Add");
    auto expected_result = std::make_shared<ov::op::v0::Result>(expected_add);
    auto expected = std::make_shared<ov::Model>(ov::ResultVector{expected_result}, ov::ParameterVector{p0, p1});

    const auto cmp = FunctionsComparator::all_flags_enabled();
    const auto res = cmp.compare(subgraph, expected);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(ExtractSubgraphTest, MultimapOverload_BranchingModel_BothBranches) {
    auto model = build_branch_model();

    // Extract both activation branches but not Mul: inputs at Relu.0 and Sigmoid.0, outputs at Relu.0 and Sigmoid.0
    const std::multimap<std::string, size_t> inputs_map = {{"Mul", 0}, {"Mul", 1}};
    const std::multimap<std::string, size_t> outputs_map = {{"Mul", 0}};

    auto subgraph = op_util::extract_subgraph(model, inputs_map, outputs_map);

    auto p0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3});
    auto p1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3});
    auto expected_mul = std::make_shared<ov::op::v1::Multiply>(p0, p1);
    expected_mul->set_friendly_name("Mul");
    auto expected_result = std::make_shared<ov::op::v0::Result>(expected_mul);
    auto expected = std::make_shared<ov::Model>(ov::ResultVector{expected_result}, ov::ParameterVector{p0, p1});

    const auto cmp = FunctionsComparator::all_flags_enabled();
    const auto res = cmp.compare(subgraph, expected);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(ExtractSubgraphTest, MultimapOverload_BranchingModel_SingleBranch) {
    auto model = build_branch_model();

    // Extract only the Relu branch up to Mul input
    const std::multimap<std::string, size_t> inputs_map = {{"Relu", 0}};
    const std::multimap<std::string, size_t> outputs_map = {{"Relu", 0}};

    auto subgraph = op_util::extract_subgraph(model, inputs_map, outputs_map);

    auto expected_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3});
    auto expected_relu = std::make_shared<ov::op::v0::Relu>(expected_param);
    expected_relu->set_friendly_name("Relu");
    auto expected_result = std::make_shared<ov::op::v0::Result>(expected_relu);
    auto expected = std::make_shared<ov::Model>(ov::ResultVector{expected_result}, ov::ParameterVector{expected_param});

    const auto cmp = FunctionsComparator::all_flags_enabled();
    const auto res = cmp.compare(subgraph, expected);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(ExtractSubgraphTest, MultimapOverload_UnknownInputNameThrows) {
    auto model = build_linear_model();

    const std::multimap<std::string, size_t> inputs_map = {{"NonExistentNode", 0}};
    const std::multimap<std::string, size_t> outputs_map = {{"Add", 0}};

    EXPECT_THROW(op_util::extract_subgraph(model, inputs_map, outputs_map), ov::Exception);
}

TEST(ExtractSubgraphTest, MultimapOverload_UnknownOutputNameThrows) {
    auto model = build_linear_model();

    const std::multimap<std::string, size_t> inputs_map = {{"Add", 0}};
    const std::multimap<std::string, size_t> outputs_map = {{"NonExistentNode", 0}};

    EXPECT_THROW(op_util::extract_subgraph(model, inputs_map, outputs_map), ov::Exception);
}

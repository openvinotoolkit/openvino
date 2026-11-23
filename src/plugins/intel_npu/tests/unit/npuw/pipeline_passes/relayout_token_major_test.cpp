// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/relayout_token_major.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"

namespace {

// A token-major trunk as the PA model has it: a weight MatMul over
// [tokens, 1, hidden], elementwise ops on it, and the RoPE position outer
// product, a MatMul with K = 1 between two activations.
std::shared_ptr<ov::Model> make_token_major_model(std::size_t tokens) {
    const auto t = static_cast<int64_t>(tokens);
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{t, 1, 16});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{t, 1, 12});
    auto freq = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{t, 4, 1});
    auto pos = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{t, 1, 1});
    std::vector<float> w(12 * 16), b(12);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : w) {
        v = dist(rng);
    }
    for (auto& v : b) {
        v = dist(rng);
    }
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{12, 16}, w);
    auto linear = std::make_shared<ov::op::v0::MatMul>(x, weights, false, true);
    auto bias = std::make_shared<ov::op::v1::Add>(linear, ov::op::v0::Constant::create(ov::element::f32, {12}, b));
    auto gate = std::make_shared<ov::op::v1::Multiply>(bias, y);
    auto rope = std::make_shared<ov::op::v0::MatMul>(freq, pos);
    return std::make_shared<ov::Model>(ov::OutputVector{gate, rope}, ov::ParameterVector{x, y, freq, pos});
}

ov::TensorVector evaluate(const std::shared_ptr<ov::Model>& model) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    ov::TensorVector inputs;
    for (const auto& param : model->get_parameters()) {
        ov::Tensor tensor(ov::element::f32, param->get_shape());
        for (std::size_t i = 0; i < tensor.get_size(); ++i) {
            tensor.data<float>()[i] = dist(rng);
        }
        inputs.push_back(tensor);
    }
    ov::TensorVector outputs;
    for (const auto& result : model->outputs()) {
        outputs.emplace_back(ov::element::f32, result.get_shape());
    }
    EXPECT_TRUE(model->evaluate(outputs, inputs));
    return outputs;
}

void relayout_token_major(const std::shared_ptr<ov::Model>& model, std::size_t tokens) {
    ov::pass::Manager manager;
    manager.register_pass<ov::npuw::RelayoutTokenMajor>(tokens);
    manager.run_passes(model);
}

TEST(RelayoutTokenMajorTest, MovesTheTokenCountOutOfTheBatchPosition) {
    constexpr std::size_t tokens = 8;
    const auto reference = make_token_major_model(tokens);
    auto model = reference->clone();
    relayout_token_major(model, tokens);

    std::size_t matmuls = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v0::MatMul>(node) && !ov::is_type<ov::op::v1::Multiply>(node) &&
            !ov::is_type<ov::op::v1::Add>(node)) {
            continue;
        }
        matmuls += ov::is_type<ov::op::v0::MatMul>(node) ? 1 : 0;
        for (const auto& input : node->inputs()) {
            const auto& shape = input.get_shape();
            EXPECT_FALSE(shape.size() >= 3 && shape[0] == tokens) << node << " input " << shape;
        }
    }
    // The outer product became a Multiply; the weight MatMul stays.
    EXPECT_EQ(matmuls, 1u);

    const auto expected = evaluate(reference);
    const auto actual = evaluate(model);
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(expected[i].get_shape(), actual[i].get_shape());
        for (std::size_t j = 0; j < expected[i].get_size(); ++j) {
            EXPECT_FLOAT_EQ(expected[i].data<float>()[j], actual[i].data<float>()[j]) << "output " << i << " at " << j;
        }
    }
}

// A dimension that is 1 in the token-major input but not in the output stays:
// an operand broadcasts into it, or a MatMul contracts over it.
TEST(RelayoutTokenMajorTest, KeepsADimensionThatIsNotOneInTheOutput) {
    constexpr std::size_t tokens = 8;
    auto x =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{static_cast<int64_t>(tokens), 4, 1});
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, std::vector<float>{1.f, 2.f, 3.f, 4.f});
    auto broadcast = std::make_shared<ov::op::v1::Multiply>(x, scale);
    auto row = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 8}, std::vector<float>(8, 0.5f));
    auto outer = std::make_shared<ov::op::v0::MatMul>(x, row);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{broadcast, outer}, ov::ParameterVector{x});
    const auto ops = model->get_ops().size();
    relayout_token_major(model, tokens);
    EXPECT_EQ(model->get_ops().size(), ops);
    EXPECT_EQ(broadcast->get_input_shape(0), (ov::Shape{tokens, 4, 1}));
    EXPECT_EQ(outer->get_input_shape(0), (ov::Shape{tokens, 4, 1}));
}

// A linear layer whose only dimension of 1 is the one the MatMul contracts
// has no dimension to give up for the batch.
TEST(RelayoutTokenMajorTest, KeepsTheDimensionALinearContracts) {
    constexpr std::size_t tokens = 8;
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 4, 1});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0.5f});
    auto linear = std::make_shared<ov::op::v0::MatMul>(x, weights, false, true);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{linear}, ov::ParameterVector{x});

    const auto ops = model->get_ops().size();
    ASSERT_NO_THROW(relayout_token_major(model, tokens));
    EXPECT_EQ(model->get_ops().size(), ops);
    EXPECT_EQ(model->output(0).get_shape(), (ov::Shape{8, 4, 1}));
}

TEST(RelayoutTokenMajorTest, LeavesTheOneTokenVariantAlone) {
    auto model = make_token_major_model(1);
    const auto ops = model->get_ops().size();
    relayout_token_major(model, 1);
    EXPECT_EQ(model->get_ops().size(), ops);
}

// A rewritten op that is a model output hands its names over to the Reshape
// that restores its shape.
TEST(RelayoutTokenMajorTest, KeepsTheNamesOfAModelOutput) {
    constexpr std::size_t tokens = 8;
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 1, 16});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{12, 16}, std::vector<float>(12 * 16, 0.5f));
    auto linear = std::make_shared<ov::op::v0::MatMul>(x, weights, false, true);
    linear->set_friendly_name("linear");
    linear->output(0).get_tensor().set_names({"hidden"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{linear}, ov::ParameterVector{x});

    relayout_token_major(model, tokens);

    EXPECT_EQ(linear->get_input_shape(0), (ov::Shape{1, tokens, 16}));
    EXPECT_EQ(model->output(0).get_names(), (std::unordered_set<std::string>{"hidden"}));
    EXPECT_EQ(model->get_results()[0]->get_input_node_ptr(0)->get_friendly_name(), "linear");
    std::set<std::string> names;
    for (const auto& op : model->get_ops()) {
        EXPECT_TRUE(names.insert(op->get_friendly_name()).second) << op->get_friendly_name();
    }
}

}  // namespace

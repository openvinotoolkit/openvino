// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset6.hpp"

#include "plugin/transformations/lora_horizontal_fusion.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, LoRAHorizontalFusion_default) {
    ov::element::Type model_dt = ov::element::f16;
    {
        auto lora_input = std::make_shared<ov::op::v0::Parameter>(model_dt, ov::PartialShape{-1, -1, 2048});
        auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{2560, 2048});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale = std::make_shared<ov::op::v0::Constant>(model_dt, ov::Shape{2560, 1});
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(lora_input, weights, bias, scale);

        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {2048, 256, 256});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fc_fused, axis_const, split_const);

        auto variable_a_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_0"});
        auto variable_alpha_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_0"});
        auto variable_b_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({2048, -1}), model_dt, "var_b_0"});
        auto read_value_a_0 = std::make_shared<ov::op::v6::ReadValue>(variable_a_0);
        auto read_value_alpha_0 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_0);
        auto read_value_b_0 = std::make_shared<ov::op::v6::ReadValue>(variable_b_0);
        auto matmul1_0 = std::make_shared<ov::op::v0::MatMul>(lora_input, read_value_a_0, false, true);
        auto multiply_0 = std::make_shared<ov::op::v1::Multiply>(matmul1_0, read_value_alpha_0);
        auto matmul2_0 = std::make_shared<ov::op::v0::MatMul>(multiply_0, read_value_b_0, false, true);
        auto add_0 = std::make_shared<ov::op::v1::Add>(split->output(0), matmul2_0);

        auto variable_a_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_1"});
        auto variable_alpha_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_1"});
        auto variable_b_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_1"});
        auto read_value_a_1 = std::make_shared<ov::op::v6::ReadValue>(variable_a_1);
        auto read_value_alpha_1 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_1);
        auto read_value_b_1 = std::make_shared<ov::op::v6::ReadValue>(variable_b_1);
        auto matmul1_1 = std::make_shared<ov::op::v0::MatMul>(lora_input, read_value_a_1, false, true);
        auto multiply_1 = std::make_shared<ov::op::v1::Multiply>(matmul1_1, read_value_alpha_1);
        auto matmul2_1 = std::make_shared<ov::op::v0::MatMul>(multiply_1, read_value_b_1, false, true);
        auto add_1 = std::make_shared<ov::op::v1::Add>(split->output(1), matmul2_1);

        auto variable_a_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_2"});
        auto variable_alpha_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_2"});
        auto variable_b_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_2"});
        auto read_value_a_2 = std::make_shared<ov::op::v6::ReadValue>(variable_a_2);
        auto read_value_alpha_2 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_2);
        auto read_value_b_2 = std::make_shared<ov::op::v6::ReadValue>(variable_b_2);
        auto matmul1_2 = std::make_shared<ov::op::v0::MatMul>(lora_input, read_value_a_2, false, true);
        auto multiply_2 = std::make_shared<ov::op::v1::Multiply>(matmul1_2, read_value_alpha_2);
        auto matmul2_2 = std::make_shared<ov::op::v0::MatMul>(multiply_2, read_value_b_2, false, true);
        auto add_2 = std::make_shared<ov::op::v1::Add>(split->output(2), matmul2_2);

        auto reshape_pattern0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 32, 64});
        auto reshape_pattern1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape_pattern2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(add_0, reshape_pattern0, true);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(add_1, reshape_pattern1, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(add_2, reshape_pattern2, true);

        auto result0 = std::make_shared<ov::op::v0::Result>(reshape0);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);

        model = std::make_shared<ov::Model>(ov::NodeVector{result0, result1, result2}, ov::ParameterVector{lora_input});
        manager.register_pass<LoRAHorizontalFusion>();
    }

    {
        auto lora_input = std::make_shared<ov::op::v0::Parameter>(model_dt, ov::PartialShape{-1, -1, 2048});
        auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{2560, 2048});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale = std::make_shared<ov::op::v0::Constant>(model_dt, ov::Shape{2560, 1});
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(lora_input, weights, bias, scale);

        auto variable_a_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_0"});
        auto variable_a_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_1"});
        auto variable_a_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_2"});

        auto read_value_a_0 = std::make_shared<ov::op::v6::ReadValue>(variable_a_0);
        auto read_value_a_1 = std::make_shared<ov::op::v6::ReadValue>(variable_a_1);
        auto read_value_a_2 = std::make_shared<ov::op::v6::ReadValue>(variable_a_2);
        auto concat_variable_a = std::make_shared<ov::op::v0::Concat>(NodeVector{read_value_a_0, read_value_a_1, read_value_a_2}, 0);

        auto fused_matmul1 = std::make_shared<ov::op::v0::MatMul>(lora_input, concat_variable_a, false, true);

        auto variable_alpha_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_0"});
        auto variable_alpha_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_1"});
        auto variable_alpha_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_2"});

        auto read_value_alpha_0 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_0);
        auto read_value_alpha_1 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_1);
        auto read_value_alpha_2 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_2);
        auto concat_variable_alpha = std::make_shared<ov::op::v0::Concat>(NodeVector{read_value_alpha_0, read_value_alpha_1, read_value_alpha_2}, 1);

        auto multiply = std::make_shared<ov::op::v1::Multiply>(fused_matmul1, concat_variable_alpha);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto split = std::make_shared<ov::op::v1::Split>(multiply, split_axis, 3);

        auto variable_b_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({2048, -1}), model_dt, "var_b_0"});
        auto variable_b_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_1"});
        auto variable_b_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_2"});

        auto read_value_b_0 = std::make_shared<ov::op::v6::ReadValue>(variable_b_0);
        auto read_value_b_1 = std::make_shared<ov::op::v6::ReadValue>(variable_b_1);
        auto read_value_b_2 = std::make_shared<ov::op::v6::ReadValue>(variable_b_2);

        auto matmul2_0 = std::make_shared<ov::op::v0::MatMul>(split->output(0), read_value_b_0, false, true);
        auto matmul2_1 = std::make_shared<ov::op::v0::MatMul>(split->output(1), read_value_b_1, false, true);
        auto matmul2_2 = std::make_shared<ov::op::v0::MatMul>(split->output(2), read_value_b_2, false, true);

        auto concat_matmul2 = std::make_shared<ov::op::v0::Concat>(NodeVector{matmul2_0, matmul2_1, matmul2_2}, 2);

        auto add = std::make_shared<ov::op::v1::Add>(fc_fused, concat_matmul2);

        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {2048, 256, 256});
        auto var_split = std::make_shared<ov::op::v1::VariadicSplit>(add, axis_const, split_const);

        auto reshape_pattern0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 32, 64});
        auto reshape_pattern1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape_pattern2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(var_split->output(0), reshape_pattern0, true);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(var_split->output(1), reshape_pattern1, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(var_split->output(2), reshape_pattern2, true);

        auto result0 = std::make_shared<ov::op::v0::Result>(reshape0);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{result0, result1, result2}, ov::ParameterVector{lora_input});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, LoRAHorizontalFusion_swap_add_and_multiply_inputs) {
    ov::element::Type model_dt = ov::element::f16;
    {
        auto lora_input = std::make_shared<ov::op::v0::Parameter>(model_dt, ov::PartialShape{-1, -1, 2048});
        auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{2560, 2048});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale = std::make_shared<ov::op::v0::Constant>(model_dt, ov::Shape{2560, 1});
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(lora_input, weights, bias, scale);

        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {2048, 256, 256});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fc_fused, axis_const, split_const);

        auto variable_a_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_0"});
        auto variable_alpha_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_0"});
        auto variable_b_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({2048, -1}), model_dt, "var_b_0"});
        auto read_value_a_0 = std::make_shared<ov::op::v6::ReadValue>(variable_a_0);
        auto read_value_alpha_0 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_0);
        auto read_value_b_0 = std::make_shared<ov::op::v6::ReadValue>(variable_b_0);
        auto matmul1_0 = std::make_shared<ov::op::v0::MatMul>(lora_input, read_value_a_0, false, true);
        auto multiply_0 = std::make_shared<ov::op::v1::Multiply>(read_value_alpha_0, matmul1_0);
        auto matmul2_0 = std::make_shared<ov::op::v0::MatMul>(multiply_0, read_value_b_0, false, true);
        auto add_0 = std::make_shared<ov::op::v1::Add>(matmul2_0, split->output(0));

        auto variable_a_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_1"});
        auto variable_alpha_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_1"});
        auto variable_b_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_1"});
        auto read_value_a_1 = std::make_shared<ov::op::v6::ReadValue>(variable_a_1);
        auto read_value_alpha_1 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_1);
        auto read_value_b_1 = std::make_shared<ov::op::v6::ReadValue>(variable_b_1);
        auto matmul1_1 = std::make_shared<ov::op::v0::MatMul>(lora_input, read_value_a_1, false, true);
        auto multiply_1 = std::make_shared<ov::op::v1::Multiply>(read_value_alpha_1, matmul1_1);
        auto matmul2_1 = std::make_shared<ov::op::v0::MatMul>(multiply_1, read_value_b_1, false, true);
        auto add_1 = std::make_shared<ov::op::v1::Add>(matmul2_1, split->output(1));

        auto variable_a_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_2"});
        auto variable_alpha_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_2"});
        auto variable_b_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_2"});
        auto read_value_a_2 = std::make_shared<ov::op::v6::ReadValue>(variable_a_2);
        auto read_value_alpha_2 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_2);
        auto read_value_b_2 = std::make_shared<ov::op::v6::ReadValue>(variable_b_2);
        auto matmul1_2 = std::make_shared<ov::op::v0::MatMul>(lora_input, read_value_a_2, false, true);
        auto multiply_2 = std::make_shared<ov::op::v1::Multiply>(read_value_alpha_2, matmul1_2);
        auto matmul2_2 = std::make_shared<ov::op::v0::MatMul>(multiply_2, read_value_b_2, false, true);
        auto add_2 = std::make_shared<ov::op::v1::Add>(matmul2_2, split->output(2));

        auto reshape_pattern0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 32, 64});
        auto reshape_pattern1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape_pattern2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(add_0, reshape_pattern0, true);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(add_1, reshape_pattern1, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(add_2, reshape_pattern2, true);

        auto result0 = std::make_shared<ov::op::v0::Result>(reshape0);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);

        model = std::make_shared<ov::Model>(ov::NodeVector{result0, result1, result2}, ov::ParameterVector{lora_input});
        manager.register_pass<LoRAHorizontalFusion>();
    }

    {
        auto lora_input = std::make_shared<ov::op::v0::Parameter>(model_dt, ov::PartialShape{-1, -1, 2048});
        auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{2560, 2048});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale = std::make_shared<ov::op::v0::Constant>(model_dt, ov::Shape{2560, 1});
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(lora_input, weights, bias, scale);

        auto variable_a_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_0"});
        auto variable_a_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_1"});
        auto variable_a_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_2"});

        auto read_value_a_0 = std::make_shared<ov::op::v6::ReadValue>(variable_a_0);
        auto read_value_a_1 = std::make_shared<ov::op::v6::ReadValue>(variable_a_1);
        auto read_value_a_2 = std::make_shared<ov::op::v6::ReadValue>(variable_a_2);
        auto concat_variable_a = std::make_shared<ov::op::v0::Concat>(NodeVector{read_value_a_0, read_value_a_1, read_value_a_2}, 0);

        auto fused_matmul1 = std::make_shared<ov::op::v0::MatMul>(lora_input, concat_variable_a, false, true);

        auto variable_alpha_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_0"});
        auto variable_alpha_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_1"});
        auto variable_alpha_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_2"});

        auto read_value_alpha_0 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_0);
        auto read_value_alpha_1 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_1);
        auto read_value_alpha_2 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_2);
        auto concat_variable_alpha = std::make_shared<ov::op::v0::Concat>(NodeVector{read_value_alpha_0, read_value_alpha_1, read_value_alpha_2}, 1);

        auto multiply = std::make_shared<ov::op::v1::Multiply>(fused_matmul1, concat_variable_alpha);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto split = std::make_shared<ov::op::v1::Split>(multiply, split_axis, 3);

        auto variable_b_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({2048, -1}), model_dt, "var_b_0"});
        auto variable_b_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_1"});
        auto variable_b_2 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_2"});

        auto read_value_b_0 = std::make_shared<ov::op::v6::ReadValue>(variable_b_0);
        auto read_value_b_1 = std::make_shared<ov::op::v6::ReadValue>(variable_b_1);
        auto read_value_b_2 = std::make_shared<ov::op::v6::ReadValue>(variable_b_2);

        auto matmul2_0 = std::make_shared<ov::op::v0::MatMul>(split->output(0), read_value_b_0, false, true);
        auto matmul2_1 = std::make_shared<ov::op::v0::MatMul>(split->output(1), read_value_b_1, false, true);
        auto matmul2_2 = std::make_shared<ov::op::v0::MatMul>(split->output(2), read_value_b_2, false, true);

        auto concat_matmul2 = std::make_shared<ov::op::v0::Concat>(NodeVector{matmul2_0, matmul2_1, matmul2_2}, 2);

        auto add = std::make_shared<ov::op::v1::Add>(fc_fused, concat_matmul2);

        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {2048, 256, 256});
        auto var_split = std::make_shared<ov::op::v1::VariadicSplit>(add, axis_const, split_const);

        auto reshape_pattern0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 32, 64});
        auto reshape_pattern1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape_pattern2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(var_split->output(0), reshape_pattern0, true);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(var_split->output(1), reshape_pattern1, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(var_split->output(2), reshape_pattern2, true);

        auto result0 = std::make_shared<ov::op::v0::Result>(reshape0);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{result0, result1, result2}, ov::ParameterVector{lora_input});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, LoRAHorizontalFusion_split_two_outputs) {
    ov::element::Type model_dt = ov::element::f16;
    {
        auto lora_input = std::make_shared<ov::op::v0::Parameter>(model_dt, ov::PartialShape{-1, -1, 2048});
        auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{2304, 2048});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale = std::make_shared<ov::op::v0::Constant>(model_dt, ov::Shape{2304, 1});
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(lora_input, weights, bias, scale);

        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2048, 256});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fc_fused, axis_const, split_const);

        auto variable_a_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_0"});
        auto variable_alpha_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_0"});
        auto variable_b_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({2048, -1}), model_dt, "var_b_0"});
        auto read_value_a_0 = std::make_shared<ov::op::v6::ReadValue>(variable_a_0);
        auto read_value_alpha_0 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_0);
        auto read_value_b_0 = std::make_shared<ov::op::v6::ReadValue>(variable_b_0);
        auto matmul1_0 = std::make_shared<ov::op::v0::MatMul>(lora_input, read_value_a_0, false, true);
        auto multiply_0 = std::make_shared<ov::op::v1::Multiply>(matmul1_0, read_value_alpha_0);
        auto matmul2_0 = std::make_shared<ov::op::v0::MatMul>(multiply_0, read_value_b_0, false, true);
        auto add_0 = std::make_shared<ov::op::v1::Add>(split->output(0), matmul2_0);

        auto variable_a_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_1"});
        auto variable_alpha_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_1"});
        auto variable_b_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_1"});
        auto read_value_a_1 = std::make_shared<ov::op::v6::ReadValue>(variable_a_1);
        auto read_value_alpha_1 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_1);
        auto read_value_b_1 = std::make_shared<ov::op::v6::ReadValue>(variable_b_1);
        auto matmul1_1 = std::make_shared<ov::op::v0::MatMul>(lora_input, read_value_a_1, false, true);
        auto multiply_1 = std::make_shared<ov::op::v1::Multiply>(matmul1_1, read_value_alpha_1);
        auto matmul2_1 = std::make_shared<ov::op::v0::MatMul>(multiply_1, read_value_b_1, false, true);
        auto add_1 = std::make_shared<ov::op::v1::Add>(split->output(1), matmul2_1);

        auto reshape_pattern0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 32, 64});
        auto reshape_pattern1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(add_0, reshape_pattern0, true);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(add_1, reshape_pattern1, true);

        auto result0 = std::make_shared<ov::op::v0::Result>(reshape0);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);

        model = std::make_shared<ov::Model>(ov::NodeVector{result0, result1}, ov::ParameterVector{lora_input});
        manager.register_pass<LoRAHorizontalFusion>();
    }

    {
        auto lora_input = std::make_shared<ov::op::v0::Parameter>(model_dt, ov::PartialShape{-1, -1, 2048});
        auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{2304, 2048});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale = std::make_shared<ov::op::v0::Constant>(model_dt, ov::Shape{2304, 1});
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(lora_input, weights, bias, scale);

        auto variable_a_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_0"});
        auto variable_a_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_1"});

        auto read_value_a_0 = std::make_shared<ov::op::v6::ReadValue>(variable_a_0);
        auto read_value_a_1 = std::make_shared<ov::op::v6::ReadValue>(variable_a_1);
        auto concat_variable_a = std::make_shared<ov::op::v0::Concat>(NodeVector{read_value_a_0, read_value_a_1}, 0);

        auto fused_matmul1 = std::make_shared<ov::op::v0::MatMul>(lora_input, concat_variable_a, false, true);

        auto variable_alpha_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_0"});
        auto variable_alpha_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_1"});

        auto read_value_alpha_0 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_0);
        auto read_value_alpha_1 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_1);
        auto concat_variable_alpha = std::make_shared<ov::op::v0::Concat>(NodeVector{read_value_alpha_0, read_value_alpha_1}, 1);

        auto multiply = std::make_shared<ov::op::v1::Multiply>(fused_matmul1, concat_variable_alpha);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto split = std::make_shared<ov::op::v1::Split>(multiply, split_axis, 2);

        auto variable_b_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({2048, -1}), model_dt, "var_b_0"});
        auto variable_b_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_1"});

        auto read_value_b_0 = std::make_shared<ov::op::v6::ReadValue>(variable_b_0);
        auto read_value_b_1 = std::make_shared<ov::op::v6::ReadValue>(variable_b_1);

        auto matmul2_0 = std::make_shared<ov::op::v0::MatMul>(split->output(0), read_value_b_0, false, true);
        auto matmul2_1 = std::make_shared<ov::op::v0::MatMul>(split->output(1), read_value_b_1, false, true);

        auto concat_matmul2 = std::make_shared<ov::op::v0::Concat>(NodeVector{matmul2_0, matmul2_1}, 2);

        auto add = std::make_shared<ov::op::v1::Add>(fc_fused, concat_matmul2);

        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2048, 256});
        auto var_split = std::make_shared<ov::op::v1::VariadicSplit>(add, axis_const, split_const);

        auto reshape_pattern0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 32, 64});
        auto reshape_pattern1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(var_split->output(0), reshape_pattern0, true);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(var_split->output(1), reshape_pattern1, true);

        auto result0 = std::make_shared<ov::op::v0::Result>(reshape0);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{result0, result1}, ov::ParameterVector{lora_input});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, LoRAHorizontalFusion_multiple_split_output_users) {
    ov::element::Type model_dt = ov::element::f16;
    {
        auto lora_input = std::make_shared<ov::op::v0::Parameter>(model_dt, ov::PartialShape{-1, -1, 2048});
        auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{2304, 2048});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale = std::make_shared<ov::op::v0::Constant>(model_dt, ov::Shape{2304, 1});
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(lora_input, weights, bias, scale);

        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2048, 256});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fc_fused, axis_const, split_const);

        auto variable_a_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_0"});
        auto variable_alpha_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_0"});
        auto variable_b_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({2048, -1}), model_dt, "var_b_0"});
        auto read_value_a_0 = std::make_shared<ov::op::v6::ReadValue>(variable_a_0);
        auto read_value_alpha_0 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_0);
        auto read_value_b_0 = std::make_shared<ov::op::v6::ReadValue>(variable_b_0);
        auto matmul1_0 = std::make_shared<ov::op::v0::MatMul>(lora_input, read_value_a_0, false, true);
        auto multiply_0 = std::make_shared<ov::op::v1::Multiply>(matmul1_0, read_value_alpha_0);
        auto matmul2_0 = std::make_shared<ov::op::v0::MatMul>(multiply_0, read_value_b_0, false, true);
        auto add_0 = std::make_shared<ov::op::v1::Add>(split->output(0), matmul2_0);

        auto variable_a_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_1"});
        auto variable_alpha_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_1"});
        auto variable_b_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_1"});
        auto read_value_a_1 = std::make_shared<ov::op::v6::ReadValue>(variable_a_1);
        auto read_value_alpha_1 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_1);
        auto read_value_b_1 = std::make_shared<ov::op::v6::ReadValue>(variable_b_1);
        auto matmul1_1 = std::make_shared<ov::op::v0::MatMul>(lora_input, read_value_a_1, false, true);
        auto multiply_1 = std::make_shared<ov::op::v1::Multiply>(matmul1_1, read_value_alpha_1);
        auto matmul2_1 = std::make_shared<ov::op::v0::MatMul>(multiply_1, read_value_b_1, false, true);
        auto add_1 = std::make_shared<ov::op::v1::Add>(split->output(1), matmul2_1);

        auto reshape_pattern0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 32, 64});
        auto reshape_pattern1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(add_0, reshape_pattern0, true);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(add_1, reshape_pattern1, true);

        auto shape_of0 = std::make_shared<ov::op::v0::ShapeOf>(add_0);
        auto shape_of1 = std::make_shared<ov::op::v0::ShapeOf>(add_0);
        auto shape_of2 = std::make_shared<ov::op::v0::ShapeOf>(add_1);
        auto shape_of3 = std::make_shared<ov::op::v0::ShapeOf>(add_1);

        auto result0 = std::make_shared<ov::op::v0::Result>(reshape0);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(shape_of0);
        auto result3 = std::make_shared<ov::op::v0::Result>(shape_of1);
        auto result4 = std::make_shared<ov::op::v0::Result>(shape_of2);
        auto result5 = std::make_shared<ov::op::v0::Result>(shape_of3);

        model = std::make_shared<ov::Model>(ov::NodeVector{result0, result1, result2, result3, result4, result5}, ov::ParameterVector{lora_input});
        manager.register_pass<LoRAHorizontalFusion>();
    }

    {
        auto lora_input = std::make_shared<ov::op::v0::Parameter>(model_dt, ov::PartialShape{-1, -1, 2048});
        auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{2304, 2048});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale = std::make_shared<ov::op::v0::Constant>(model_dt, ov::Shape{2304, 1});
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(lora_input, weights, bias, scale);

        auto variable_a_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_0"});
        auto variable_a_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, 2048}), model_dt, "var_a_1"});

        auto read_value_a_0 = std::make_shared<ov::op::v6::ReadValue>(variable_a_0);
        auto read_value_a_1 = std::make_shared<ov::op::v6::ReadValue>(variable_a_1);
        auto concat_variable_a = std::make_shared<ov::op::v0::Concat>(NodeVector{read_value_a_0, read_value_a_1}, 0);

        auto fused_matmul1 = std::make_shared<ov::op::v0::MatMul>(lora_input, concat_variable_a, false, true);

        auto variable_alpha_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_0"});
        auto variable_alpha_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), model_dt, "var_alpha_1"});

        auto read_value_alpha_0 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_0);
        auto read_value_alpha_1 = std::make_shared<ov::op::v6::ReadValue>(variable_alpha_1);
        auto concat_variable_alpha = std::make_shared<ov::op::v0::Concat>(NodeVector{read_value_alpha_0, read_value_alpha_1}, 1);

        auto multiply = std::make_shared<ov::op::v1::Multiply>(fused_matmul1, concat_variable_alpha);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto split = std::make_shared<ov::op::v1::Split>(multiply, split_axis, 2);

        auto variable_b_0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({2048, -1}), model_dt, "var_b_0"});
        auto variable_b_1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({256, -1}), model_dt, "var_b_1"});

        auto read_value_b_0 = std::make_shared<ov::op::v6::ReadValue>(variable_b_0);
        auto read_value_b_1 = std::make_shared<ov::op::v6::ReadValue>(variable_b_1);

        auto matmul2_0 = std::make_shared<ov::op::v0::MatMul>(split->output(0), read_value_b_0, false, true);
        auto matmul2_1 = std::make_shared<ov::op::v0::MatMul>(split->output(1), read_value_b_1, false, true);

        auto concat_matmul2 = std::make_shared<ov::op::v0::Concat>(NodeVector{matmul2_0, matmul2_1}, 2);

        auto add = std::make_shared<ov::op::v1::Add>(fc_fused, concat_matmul2);

        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2048, 256});
        auto var_split = std::make_shared<ov::op::v1::VariadicSplit>(add, axis_const, split_const);

        auto reshape_pattern0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 32, 64});
        auto reshape_pattern1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 4, 64});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(var_split->output(0), reshape_pattern0, true);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(var_split->output(1), reshape_pattern1, true);

        auto shape_of0 = std::make_shared<ov::op::v0::ShapeOf>(var_split->output(0));
        auto shape_of1 = std::make_shared<ov::op::v0::ShapeOf>(var_split->output(0));
        auto shape_of2 = std::make_shared<ov::op::v0::ShapeOf>(var_split->output(1));
        auto shape_of3 = std::make_shared<ov::op::v0::ShapeOf>(var_split->output(1));

        auto result0 = std::make_shared<ov::op::v0::Result>(reshape0);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(shape_of0);
        auto result3 = std::make_shared<ov::op::v0::Result>(shape_of1);
        auto result4 = std::make_shared<ov::op::v0::Result>(shape_of2);
        auto result5 = std::make_shared<ov::op::v0::Result>(shape_of3);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{result0, result1, result2, result3, result4, result5}, ov::ParameterVector{lora_input});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov

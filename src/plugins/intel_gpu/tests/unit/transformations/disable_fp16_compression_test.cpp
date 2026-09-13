// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include <openvino/core/model.hpp>
#include <transformations/rt_info/disable_precision_conversion.hpp>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "plugin/transformations/disable_fp16_comp_direct_multiply_sin_cos.hpp"
#include "plugin/transformations/disable_fp16_comp_gated_residual.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

struct QwenImageGate {
    std::shared_ptr<ov::op::v0::Parameter> input;
    std::shared_ptr<ov::op::v0::Unsqueeze> output;
};

QwenImageGate make_qwen_image_gate(const ov::element::Type& type = ov::element::f32) {
    auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{1, 1024});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto outer_lengths = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {384, 640});
    auto outer_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis, outer_lengths);
    auto inner_lengths = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {128, 128, 128});
    auto inner_split = std::make_shared<ov::op::v1::VariadicSplit>(outer_split->output(0), axis, inner_lengths);
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto output = std::make_shared<ov::op::v0::Unsqueeze>(inner_split->output(2), unsqueeze_axis);
    return {input, output};
}

struct QwenImageBranch {
    std::shared_ptr<ov::op::v0::Parameter> input;
    std::shared_ptr<ov::op::v0::Parameter> weights;
    std::shared_ptr<ov::op::v0::MatMul> matmul;
    std::shared_ptr<ov::op::v1::Add> output;
};

QwenImageBranch make_qwen_image_branch(const ov::element::Type& type = ov::element::f32) {
    auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{1, 32, 64});
    auto weights = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{64, 128});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights);
    auto bias = ov::op::v0::Constant::create(type, ov::Shape{128}, {0});
    auto output = std::make_shared<ov::op::v1::Add>(matmul, bias);
    return {input, weights, matmul, output};
}

std::shared_ptr<ov::Model> make_qwen_image_model(bool is_reference = false,
                                                 const ov::element::Type& type = ov::element::f32,
                                                 bool multiply_first = false,
                                                 bool scale_residual = false) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{1, 32, 128});
    std::shared_ptr<ov::Node> residual_input = residual;
    if (scale_residual) {
        auto residual_scale = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
        residual_input = std::make_shared<ov::op::v1::Multiply>(residual, residual_scale);
    }

    auto gate = make_qwen_image_gate(type);
    auto branch = make_qwen_image_branch(type);
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate.output, branch.output);
    auto add = multiply_first ? std::make_shared<ov::op::v1::Add>(gated_branch, residual_input)
                              : std::make_shared<ov::op::v1::Add>(residual_input, gated_branch);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn = std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    if (is_reference) {
        for (const auto& node : ov::NodeVector{branch.matmul, branch.output, gate.output, gated_branch, add, mvn})
            ov::disable_conversion(node, ov::element::f16);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{mvn},
                                       ov::ParameterVector{residual, gate.input, branch.input, branch.weights});
}

std::shared_ptr<ov::Model> make_flux_gate_model() {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gate_lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 384});
    auto gate_rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 384});
    auto gate_source = std::make_shared<ov::op::v1::Add>(gate_lhs, gate_rhs);
    auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto lengths = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {128, 128, 128});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(gate_source, axis, lengths);
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto gate = std::make_shared<ov::op::v0::Unsqueeze>(split->output(0), unsqueeze_axis);
    auto branch = make_qwen_image_branch();
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate, branch.output);
    auto add = std::make_shared<ov::op::v1::Add>(residual, gated_branch);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn = std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    return std::make_shared<ov::Model>(
        ov::OutputVector{mvn},
        ov::ParameterVector{residual, gate_lhs, gate_rhs, branch.input, branch.weights});
}

std::shared_ptr<ov::Model> make_mvn_model(bool with_add) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    std::shared_ptr<ov::Node> mvn_input = lhs;
    ov::ParameterVector parameters{lhs};
    if (with_add) {
        auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
        mvn_input = std::make_shared<ov::op::v1::Add>(lhs, rhs);
        parameters.push_back(rhs);
    }
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn = std::make_shared<ov::op::v6::MVN>(mvn_input, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);
    return std::make_shared<ov::Model>(ov::OutputVector{mvn}, parameters);
}

std::shared_ptr<ov::Model> make_direct_sin_cos_model(bool is_reference = false,
                                                     const ov::element::Type& type = ov::element::f32,
                                                     size_t cos_count = 1) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{1, 32});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{1, 32});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
    auto sin = std::make_shared<ov::op::v0::Sin>(multiply);

    ov::OutputVector outputs{sin};
    ov::NodeVector protected_nodes{lhs, rhs, multiply, sin};
    for (size_t i = 0; i < cos_count; ++i) {
        auto cos = std::make_shared<ov::op::v0::Cos>(multiply);
        outputs.push_back(cos);
        protected_nodes.push_back(cos);
    }

    if (is_reference) {
        for (const auto& node : protected_nodes)
            ov::disable_conversion(node, ov::element::f16);
    }

    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{lhs, rhs});
}

class DisableFP16CompForQwenImageGatedResidualTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<DisableFP16CompForQwenImageGatedResidualPattern>();
    }
};

class DisableFP16CompForDirectMultiplySinCosTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<DisableFP16CompForDirectMultiplySinCos>();
    }
};

}  // namespace

TEST_F(DisableFP16CompForQwenImageGatedResidualTest, Positive) {
    model = make_qwen_image_model();
    model_ref = make_qwen_image_model(true);
}

TEST_F(DisableFP16CompForQwenImageGatedResidualTest, FluxGate_NoOp) {
    model = make_flux_gate_model();
}

TEST_F(DisableFP16CompForQwenImageGatedResidualTest, Negative) {
    model = make_mvn_model(false);
}

TEST_F(DisableFP16CompForQwenImageGatedResidualTest, AddWithoutMultiply_NoOp) {
    model = make_mvn_model(true);
}

TEST_F(DisableFP16CompForQwenImageGatedResidualTest, MultiplyFirst_Positive) {
    model = make_qwen_image_model(false, ov::element::f32, true);
    model_ref = make_qwen_image_model(true, ov::element::f32, true);
}

TEST_F(DisableFP16CompForQwenImageGatedResidualTest, TwoMultiplyInputs) {
    model = make_qwen_image_model(false, ov::element::f32, false, true);
    model_ref = make_qwen_image_model(true, ov::element::f32, false, true);
}

TEST_F(DisableFP16CompForQwenImageGatedResidualTest, FP16_NoOp) {
    model = make_qwen_image_model(false, ov::element::f16);
}

TEST_F(DisableFP16CompForDirectMultiplySinCosTest, Positive) {
    model = make_direct_sin_cos_model();
    model_ref = make_direct_sin_cos_model(true);
}

TEST_F(DisableFP16CompForDirectMultiplySinCosTest, WithoutCos_NoOp) {
    model = make_direct_sin_cos_model(false, ov::element::f32, 0);
}

TEST_F(DisableFP16CompForDirectMultiplySinCosTest, FP16_NoOp) {
    model = make_direct_sin_cos_model(false, ov::element::f16);
}

TEST_F(DisableFP16CompForDirectMultiplySinCosTest, MultipleCos) {
    model = make_direct_sin_cos_model(false, ov::element::f32, 2);
    model_ref = make_direct_sin_cos_model(true, ov::element::f32, 2);
}
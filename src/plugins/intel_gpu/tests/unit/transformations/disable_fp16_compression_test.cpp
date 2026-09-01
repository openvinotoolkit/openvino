// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/rt_info/disable_precision_conversion.hpp>
#include <transformations/utils/utils.hpp>

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
    std::shared_ptr<ov::op::v1::Add> output;
};

QwenImageBranch make_qwen_image_branch(const ov::element::Type& type = ov::element::f32) {
    auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{1, 32, 64});
    auto weights = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{64, 128});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights);
    auto bias = ov::op::v0::Constant::create(type, ov::Shape{128}, {0});
    auto output = std::make_shared<ov::op::v1::Add>(matmul, bias);
    return {input, weights, output};
}

TEST(TransformationTests, DisableFP16CompForQwenImageGatedResidual_Positive) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gate = make_qwen_image_gate();
    auto branch = make_qwen_image_branch();
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate.output, branch.output);
    auto add = std::make_shared<ov::op::v1::Add>(residual, gated_branch);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{mvn},
        ov::ParameterVector{residual, gate.input, branch.input, branch.weights});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForQwenImageGatedResidualPattern>();
    manager.run_passes(model);

    ASSERT_TRUE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForQwenImageGatedResidual_FluxGate_NoOp) {
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
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{mvn},
        ov::ParameterVector{residual, gate_lhs, gate_rhs, branch.input, branch.weights});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForQwenImageGatedResidualPattern>();
    manager.run_passes(model);

    EXPECT_FALSE(ov::is_conversion_disabled(gated_branch, ov::element::f16));
    EXPECT_FALSE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForQwenImageGatedResidual_DisabledByPassConfig) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gate = make_qwen_image_gate();
    auto branch = make_qwen_image_branch();
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate.output, branch.output);
    auto add = std::make_shared<ov::op::v1::Add>(residual, gated_branch);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{mvn},
        ov::ParameterVector{residual, gate.input, branch.input, branch.weights});
    ov::pass::Manager manager;
    manager.get_pass_config()->disable<DisableFP16CompForQwenImageGatedResidualPattern>();
    manager.register_pass<DisableFP16CompForQwenImageGatedResidualPattern>();
    manager.run_passes(model);

    EXPECT_FALSE(ov::is_conversion_disabled(gated_branch, ov::element::f16));
    EXPECT_FALSE(ov::is_conversion_disabled(add, ov::element::f16));
    EXPECT_FALSE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForQwenImageGatedResidual_ConvertPrecision) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gate = make_qwen_image_gate();
    auto branch_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 64});
    auto branch_weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{64, 128});
    auto branch_matmul = std::make_shared<ov::op::v0::MatMul>(branch_input, branch_weights);
    branch_matmul->set_friendly_name("branch_matmul");
    auto branch_bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {0});
    auto branch = std::make_shared<ov::op::v1::Add>(branch_matmul, branch_bias);
    branch->set_friendly_name("branch");
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate.output, branch);
    gated_branch->set_friendly_name("gated_branch");
    auto add = std::make_shared<ov::op::v1::Add>(residual, gated_branch);
    add->set_friendly_name("residual_add");
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);
    mvn->set_friendly_name("mvn");

    auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn},
                                             ov::ParameterVector{residual, gate.input, branch_input, branch_weights});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForQwenImageGatedResidualPattern>();
    precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map, type_to_fuse_map{}, true, false, true);
    manager.run_passes(model);

    size_t checked_nodes = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_friendly_name() == "branch_matmul" || op->get_friendly_name() == "branch" ||
            op->get_friendly_name() == "gated_branch" || op->get_friendly_name() == "residual_add" ||
            op->get_friendly_name() == "mvn") {
            EXPECT_EQ(op->get_output_element_type(0), ov::element::f32) << op->get_friendly_name();
            ++checked_nodes;
        }
    }
    EXPECT_EQ(checked_nodes, 5);
}

TEST(TransformationTests, DisableFP16CompForQwenImageGatedResidual_Negative) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(input, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn}, ov::ParameterVector{input});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForQwenImageGatedResidualPattern>();
    manager.run_passes(model);

    ASSERT_FALSE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForQwenImageGatedResidual_AddWithoutMultiply_NoOp) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn}, ov::ParameterVector{lhs, rhs});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForQwenImageGatedResidualPattern>();
    manager.run_passes(model);

    ASSERT_FALSE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForQwenImageGatedResidual_MultiplyFirst_Positive) {
    auto gate = make_qwen_image_gate();
    auto branch = make_qwen_image_branch();
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate.output, branch.output);
    auto add = std::make_shared<ov::op::v1::Add>(gated_branch, residual);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{mvn},
        ov::ParameterVector{gate.input, branch.input, branch.weights, residual});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForQwenImageGatedResidualPattern>();
    manager.run_passes(model);

    ASSERT_TRUE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForQwenImageGatedResidual_TwoMultiplyInputs) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto residual_scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.5f});
    auto scaled_residual = std::make_shared<ov::op::v1::Multiply>(residual, residual_scale);
    scaled_residual->set_friendly_name("scaled_residual");

    auto gate = make_qwen_image_gate();
    auto branch_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 64});
    auto branch_weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{64, 128});
    auto branch_matmul = std::make_shared<ov::op::v0::MatMul>(branch_input, branch_weights);
    branch_matmul->set_friendly_name("branch_matmul");
    auto branch_bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {0});
    auto branch = std::make_shared<ov::op::v1::Add>(branch_matmul, branch_bias);
    branch->set_friendly_name("branch");
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate.output, branch);
    gated_branch->set_friendly_name("gated_branch");

    auto add = std::make_shared<ov::op::v1::Add>(scaled_residual, gated_branch);
    add->set_friendly_name("residual_add");
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);
    mvn->set_friendly_name("mvn");

    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{mvn},
        ov::ParameterVector{residual, gate.input, branch_input, branch_weights});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForQwenImageGatedResidualPattern>();
    precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map, type_to_fuse_map{}, true, false, true);
    manager.run_passes(model);

    for (const auto& op : {std::static_pointer_cast<ov::Node>(branch_matmul),
                           std::static_pointer_cast<ov::Node>(branch),
                           std::static_pointer_cast<ov::Node>(gated_branch),
                           std::static_pointer_cast<ov::Node>(add),
                           std::static_pointer_cast<ov::Node>(mvn)}) {
        EXPECT_EQ(op->get_output_element_type(0), ov::element::f32) << op->get_friendly_name();
    }
}

TEST(TransformationTests, DisableFP16CompForQwenImageGatedResidual_FP16_NoOp) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 32, 128});
    auto gate = make_qwen_image_gate(ov::element::f16);
    auto branch = make_qwen_image_branch(ov::element::f16);
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate.output, branch.output);
    auto add = std::make_shared<ov::op::v1::Add>(residual, gated_branch);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{mvn},
        ov::ParameterVector{residual, gate.input, branch.input, branch.weights});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForQwenImageGatedResidualPattern>();
    manager.run_passes(model);

    ASSERT_FALSE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForDirectMultiplySinCos_Positive) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
    auto sin = std::make_shared<ov::op::v0::Sin>(multiply);
    auto cos = std::make_shared<ov::op::v0::Cos>(multiply);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{sin, cos}, ov::ParameterVector{lhs, rhs});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForDirectMultiplySinCos>();
    precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map, type_to_fuse_map{}, true, false, true);
    manager.run_passes(model);

    EXPECT_TRUE(ov::is_conversion_disabled(lhs, ov::element::f16));
    EXPECT_TRUE(ov::is_conversion_disabled(rhs, ov::element::f16));
    EXPECT_TRUE(ov::is_conversion_disabled(multiply, ov::element::f16));
    EXPECT_TRUE(ov::is_conversion_disabled(sin, ov::element::f16));
    EXPECT_TRUE(ov::is_conversion_disabled(cos, ov::element::f16));
    EXPECT_EQ(multiply->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(sin->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(cos->get_output_element_type(0), ov::element::f32);
}

TEST(TransformationTests, DisableFP16CompForDirectMultiplySinCos_WithoutCos_NoOp) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
    auto sin = std::make_shared<ov::op::v0::Sin>(multiply);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{lhs, rhs});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForDirectMultiplySinCos>();
    manager.run_passes(model);

    EXPECT_FALSE(ov::is_conversion_disabled(lhs, ov::element::f16));
    EXPECT_FALSE(ov::is_conversion_disabled(rhs, ov::element::f16));
    EXPECT_FALSE(ov::is_conversion_disabled(multiply, ov::element::f16));
    EXPECT_FALSE(ov::is_conversion_disabled(sin, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForDirectMultiplySinCos_FP16_NoOp) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 32});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 32});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
    auto sin = std::make_shared<ov::op::v0::Sin>(multiply);
    auto cos = std::make_shared<ov::op::v0::Cos>(multiply);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{sin, cos}, ov::ParameterVector{lhs, rhs});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForDirectMultiplySinCos>();
    manager.run_passes(model);

    EXPECT_FALSE(ov::is_conversion_disabled(multiply, ov::element::f16));
    EXPECT_FALSE(ov::is_conversion_disabled(sin, ov::element::f16));
    EXPECT_FALSE(ov::is_conversion_disabled(cos, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForDirectMultiplySinCos_MultipleCos) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
    auto sin = std::make_shared<ov::op::v0::Sin>(multiply);
    auto cos_1 = std::make_shared<ov::op::v0::Cos>(multiply);
    auto cos_2 = std::make_shared<ov::op::v0::Cos>(multiply);

    auto model =
        std::make_shared<ov::Model>(ov::OutputVector{sin, cos_1, cos_2}, ov::ParameterVector{lhs, rhs});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForDirectMultiplySinCos>();
    manager.run_passes(model);

    EXPECT_TRUE(ov::is_conversion_disabled(multiply, ov::element::f16));
    EXPECT_TRUE(ov::is_conversion_disabled(sin, ov::element::f16));
    EXPECT_TRUE(ov::is_conversion_disabled(cos_1, ov::element::f16));
    EXPECT_TRUE(ov::is_conversion_disabled(cos_2, ov::element::f16));
}

}  // namespace
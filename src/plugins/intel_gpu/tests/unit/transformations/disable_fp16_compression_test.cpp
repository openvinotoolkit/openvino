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
#include "plugin/transformations/disable_fp16_comp_direct_multiply_sin_cos.hpp"
#include "plugin/transformations/disable_fp16_comp_gated_residual.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

TEST(TransformationTests, DisableFP16CompForGatedResidual_Positive) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gate = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 128});
    auto branch = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate, branch);
    auto add = std::make_shared<ov::op::v1::Add>(residual, gated_branch);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn}, ov::ParameterVector{residual, gate, branch});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGatedResidualPattern>();
    manager.run_passes(model);

    ASSERT_TRUE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForGatedResidual_DisabledByPassConfig) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gate = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 128});
    auto branch = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate, branch);
    auto add = std::make_shared<ov::op::v1::Add>(residual, gated_branch);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn}, ov::ParameterVector{residual, gate, branch});
    ov::pass::Manager manager;
    manager.get_pass_config()->disable<DisableFP16CompForGatedResidualPattern>();
    manager.register_pass<DisableFP16CompForGatedResidualPattern>();
    manager.run_passes(model);

    EXPECT_FALSE(ov::is_conversion_disabled(gated_branch, ov::element::f16));
    EXPECT_FALSE(ov::is_conversion_disabled(add, ov::element::f16));
    EXPECT_FALSE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForGatedResidual_ConvertPrecision) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gate = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 128});
    auto branch_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 64});
    auto branch_weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{64, 128});
    auto branch_matmul = std::make_shared<ov::op::v0::MatMul>(branch_input, branch_weights);
    branch_matmul->set_friendly_name("branch_matmul");
    auto branch_bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {0});
    auto branch = std::make_shared<ov::op::v1::Add>(branch_matmul, branch_bias);
    branch->set_friendly_name("branch");
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate, branch);
    gated_branch->set_friendly_name("gated_branch");
    auto add = std::make_shared<ov::op::v1::Add>(residual, gated_branch);
    add->set_friendly_name("residual_add");
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);
    mvn->set_friendly_name("mvn");

    auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn},
                                             ov::ParameterVector{residual, gate, branch_input, branch_weights});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGatedResidualPattern>();
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

TEST(TransformationTests, DisableFP16CompForGatedResidual_Negative) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(input, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn}, ov::ParameterVector{input});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGatedResidualPattern>();
    manager.run_passes(model);

    ASSERT_FALSE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForGatedResidual_AddWithoutMultiply_NoOp) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn}, ov::ParameterVector{lhs, rhs});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGatedResidualPattern>();
    manager.run_passes(model);

    ASSERT_FALSE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForGatedResidual_MultiplyFirst_Positive) {
    auto gate = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 128});
    auto branch = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate, branch);
    auto add = std::make_shared<ov::op::v1::Add>(gated_branch, residual);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn}, ov::ParameterVector{gate, branch, residual});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGatedResidualPattern>();
    manager.run_passes(model);

    ASSERT_TRUE(ov::is_conversion_disabled(mvn, ov::element::f16));
}

TEST(TransformationTests, DisableFP16CompForGatedResidual_TwoMultiplyInputs) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto residual_scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.5f});
    auto scaled_residual = std::make_shared<ov::op::v1::Multiply>(residual, residual_scale);
    scaled_residual->set_friendly_name("scaled_residual");

    auto gate = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 128});
    auto branch_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 64});
    auto branch_weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{64, 128});
    auto branch_matmul = std::make_shared<ov::op::v0::MatMul>(branch_input, branch_weights);
    branch_matmul->set_friendly_name("branch_matmul");
    auto branch_bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {0});
    auto branch = std::make_shared<ov::op::v1::Add>(branch_matmul, branch_bias);
    branch->set_friendly_name("branch");
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate, branch);
    gated_branch->set_friendly_name("gated_branch");

    auto add = std::make_shared<ov::op::v1::Add>(scaled_residual, gated_branch);
    add->set_friendly_name("residual_add");
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);
    mvn->set_friendly_name("mvn");

    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{mvn},
        ov::ParameterVector{residual, gate, branch_input, branch_weights});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGatedResidualPattern>();
    precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map, type_to_fuse_map{}, true, false, true);
    manager.run_passes(model);

    for (const auto& op : {std::static_pointer_cast<ov::Node>(scaled_residual),
                           std::static_pointer_cast<ov::Node>(branch_matmul),
                           std::static_pointer_cast<ov::Node>(branch),
                           std::static_pointer_cast<ov::Node>(gated_branch),
                           std::static_pointer_cast<ov::Node>(add),
                           std::static_pointer_cast<ov::Node>(mvn)}) {
        EXPECT_EQ(op->get_output_element_type(0), ov::element::f32) << op->get_friendly_name();
    }
}

TEST(TransformationTests, DisableFP16CompForGatedResidual_FP16_NoOp) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 32, 128});
    auto gate = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 1, 128});
    auto branch = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 32, 128});
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate, branch);
    auto add = std::make_shared<ov::op::v1::Add>(residual, gated_branch);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn}, ov::ParameterVector{residual, gate, branch});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGatedResidualPattern>();
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
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/rt_info/disable_precision_conversion.hpp>
#include <transformations/utils/utils.hpp>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "ov_ops/rms.hpp"
#include "plugin/transformations/disable_fp16_compression.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

namespace rms {
const std::string name_rms_1 = "rms_1";
const std::string name_rms_2 = "rms_2";

// This model creates the exact pattern that DisableFP16CompForGemma3RMSPattern is looking for.
// (Add, RMS) -> Add -> RMS
std::shared_ptr<ov::Model> create_model_to_match(bool use_convert = false) {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});
    auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});
    auto input3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});

    // Pattern part 1: add_m
    auto add_m = std::make_shared<ov::op::v1::Add>(input1, input2);

    // Pattern part 2: rms_post_m
    std::shared_ptr<ov::Node> rms_const_or_convert_1;
    if (use_convert) {
        auto const_node_1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{128}, {1.0f});
        rms_const_or_convert_1 = std::make_shared<ov::op::v0::Convert>(const_node_1, ov::element::f32);
    } else {
        rms_const_or_convert_1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    }
    auto rms_post_m = std::make_shared<ov::op::internal::RMS>(input3, rms_const_or_convert_1, 1e-5);
    rms_post_m->set_friendly_name(name_rms_1);

    // Pattern part 3: add_1_m
    auto add_1_m = std::make_shared<ov::op::v1::Add>(add_m, rms_post_m);

    // Pattern part 4: rms_m
    std::shared_ptr<ov::Node> rms_const_or_convert_2;
    if (use_convert) {
        auto const_node_2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{128}, {1.0f});
        rms_const_or_convert_2 = std::make_shared<ov::op::v0::Convert>(const_node_2, ov::element::f32);
    } else {
        rms_const_or_convert_2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    }
    auto rms_m = std::make_shared<ov::op::internal::RMS>(add_1_m, rms_const_or_convert_2, 1e-5);
    rms_m->set_friendly_name(name_rms_2);

    return std::make_shared<ov::Model>(ov::OutputVector{rms_m}, ov::ParameterVector{input1, input2, input3});
}

// This model has a similar structure but doesn't match the specific pattern.
std::shared_ptr<ov::Model> create_model_not_to_match() {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});

    auto rms_const_1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    auto rms_1 = std::make_shared<ov::op::internal::RMS>(input1, rms_const_1, 1e-5);
    rms_1->set_friendly_name(name_rms_1);

    auto some_other_op = std::make_shared<ov::op::v1::Add>(
        rms_1,
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}));

    auto rms_const_2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    auto rms_2 = std::make_shared<ov::op::internal::RMS>(some_other_op, rms_const_2, 1e-5);
    rms_2->set_friendly_name(name_rms_2);

    return std::make_shared<ov::Model>(ov::OutputVector{rms_2}, ov::ParameterVector{input1});
}

void run_test(std::shared_ptr<ov::Model> model,
              const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGemma3RMSPattern>();

    precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map);

    manager.run_passes(model);

    for (const auto& op : model->get_ops()) {
        auto it = expected_fp16_disabled_status.find(op->get_friendly_name());
        if (it != expected_fp16_disabled_status.end()) {
            bool expected_status = it->second;
            if (expected_status) {
                ASSERT_TRUE(ov::is_conversion_disabled(op, ov::element::f16))
                    << "FP16 compression is not disabled for node: " << op->get_friendly_name();
            } else {
                ASSERT_FALSE(ov::is_conversion_disabled(op, ov::element::f16))
                    << "FP16 compression is unexpectedly disabled for node: " << op->get_friendly_name();
            }
        }
    }
}
}  // namespace rms

TEST(TransformationTests, DisableFP16CompForRMS_Positive) {
    auto model = rms::create_model_to_match();
    // In the matching pattern, both rms_1 (rms_post_m) and rms_2 (rms_m) should have FP16 compression disabled.
    std::unordered_map<std::string, bool> expected_status = {
        {rms::name_rms_1, true},
        {rms::name_rms_2, true},
    };
    rms::run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompForRMS_PositiveConvert) {
    auto model = rms::create_model_to_match(true);
    // In the matching pattern, both rms_1 (rms_post_m) and rms_2 (rms_m) should have FP16 compression disabled.
    std::unordered_map<std::string, bool> expected_status = {
        {rms::name_rms_1, true},
        {rms::name_rms_2, true},
    };
    rms::run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompForRMS_Negative) {
    auto model = rms::create_model_not_to_match();
    // In the non-matching model, no RMS node should have FP16 compression disabled by the pass.
    std::unordered_map<std::string, bool> expected_status = {
        {rms::name_rms_1, false},
        {rms::name_rms_2, false},
    };
    rms::run_test(model, expected_status);
}

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

TEST(TransformationTests, DisableFP16CompForGatedResidual_ConvertPrecision) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gate = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 128});
    auto branch = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 128});
    auto gated_branch = std::make_shared<ov::op::v1::Multiply>(gate, branch);
    gated_branch->set_friendly_name("gated_branch");
    auto add = std::make_shared<ov::op::v1::Add>(residual, gated_branch);
    add->set_friendly_name("residual_add");
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(add, axes, true, 1e-6, ov::op::MVNEpsMode::INSIDE_SQRT);
    mvn->set_friendly_name("mvn");

    auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn}, ov::ParameterVector{residual, gate, branch});
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16Compression>();
    precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map, type_to_fuse_map{}, true, false, true);
    manager.run_passes(model);

    size_t checked_nodes = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_friendly_name() == "gated_branch" || op->get_friendly_name() == "residual_add" ||
            op->get_friendly_name() == "mvn") {
            EXPECT_EQ(op->get_output_element_type(0), ov::element::f32) << op->get_friendly_name();
            ++checked_nodes;
        }
    }
    EXPECT_EQ(checked_nodes, 3);
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
    manager.register_pass<DisableFP16Compression>();
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

namespace cumsum_sin_gen {
// Friendly names used to look up the matched nodes after the pass has run.
const std::string name_cumsum_input = "cumsum_input";
const std::string name_cumsum = "cumsum";
const std::string name_mul1 = "mul1";
const std::string name_transpose2 = "transpose2";
const std::string name_mul2 = "mul2";
const std::string name_interpolate = "interpolate";
const std::string name_transpose3 = "transpose3";
const std::string name_sin = "sin";

// Build the full chain matched by DisableFP16CompCumSumSinGen:
//   producer -> CumSum -> Multiply -> Transpose -> Multiply -> Interpolate -> Transpose -> Sin
std::shared_ptr<ov::Model> create_model_to_match() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    // Extra Multiply so the test can verify the CumSum producer gets marked.
    auto producer_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {1.0f});
    auto cumsum_input = std::make_shared<ov::op::v1::Multiply>(input, producer_const);
    cumsum_input->set_friendly_name(name_cumsum_input);

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto cumsum = std::make_shared<ov::op::v0::CumSum>(cumsum_input, axis);
    cumsum->set_friendly_name(name_cumsum);

    auto mul1_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {6.2832f});
    auto mul1 = std::make_shared<ov::op::v1::Multiply>(cumsum, mul1_const);
    mul1->set_friendly_name(name_mul1);

    auto order2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose2 = std::make_shared<ov::op::v1::Transpose>(mul1, order2);
    transpose2->set_friendly_name(name_transpose2);

    auto mul2_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1.0f});
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(transpose2, mul2_const);
    mul2->set_friendly_name(name_mul2);

    ov::op::v4::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v4::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ov::op::v4::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {64, 1});
    auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {2.0f, 1.0f});
    auto interpolate = std::make_shared<ov::op::v4::Interpolate>(mul2, target_shape, scales, attrs);
    interpolate->set_friendly_name(name_interpolate);

    auto order3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose3 = std::make_shared<ov::op::v1::Transpose>(interpolate, order3);
    transpose3->set_friendly_name(name_transpose3);

    auto sin = std::make_shared<ov::op::v0::Sin>(transpose3);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

// A bare Sin (no upstream CumSum) — must not be matched.
std::shared_ptr<ov::Model> create_model_not_to_match() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    auto mul_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {1.0f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(input, mul_const);
    mul->set_friendly_name(name_mul1);

    auto sin = std::make_shared<ov::op::v0::Sin>(mul);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

// Same chain but without the second Multiply between Transpose_2 and
// Interpolate — must not match.
std::shared_ptr<ov::Model> create_model_missing_mul2() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto cumsum = std::make_shared<ov::op::v0::CumSum>(input, axis);
    cumsum->set_friendly_name(name_cumsum);

    auto mul1_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {6.2832f});
    auto mul1 = std::make_shared<ov::op::v1::Multiply>(cumsum, mul1_const);
    mul1->set_friendly_name(name_mul1);

    auto order2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose2 = std::make_shared<ov::op::v1::Transpose>(mul1, order2);
    transpose2->set_friendly_name(name_transpose2);

    // NOTE: no second Multiply here.

    ov::op::v4::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v4::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ov::op::v4::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {64, 1});
    auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {2.0f, 1.0f});
    auto interpolate = std::make_shared<ov::op::v4::Interpolate>(transpose2, target_shape, scales, attrs);
    interpolate->set_friendly_name(name_interpolate);

    auto order3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose3 = std::make_shared<ov::op::v1::Transpose>(interpolate, order3);
    transpose3->set_friendly_name(name_transpose3);

    auto sin = std::make_shared<ov::op::v0::Sin>(transpose3);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

void run_test(const std::shared_ptr<ov::Model>& model,
              const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompCumSumSinGen>();

    precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map);

    manager.run_passes(model);

    for (const auto& op : model->get_ops()) {
        auto it = expected_fp16_disabled_status.find(op->get_friendly_name());
        if (it == expected_fp16_disabled_status.end())
            continue;
        if (it->second) {
            ASSERT_TRUE(ov::is_conversion_disabled(op, ov::element::f16))
                << "FP16 compression is not disabled for node: " << op->get_friendly_name();
        } else {
            ASSERT_FALSE(ov::is_conversion_disabled(op, ov::element::f16))
                << "FP16 compression is unexpectedly disabled for node: " << op->get_friendly_name();
        }
    }
}
}  // namespace cumsum_sin_gen

TEST(TransformationTests, DisableFP16CompCumSumSinGen_Positive) {
    auto model = cumsum_sin_gen::create_model_to_match();
    // The pass marks the full 7-node chain plus the producer feeding CumSum.
    std::unordered_map<std::string, bool> expected_status = {
        {cumsum_sin_gen::name_cumsum_input, true},
        {cumsum_sin_gen::name_cumsum, true},
        {cumsum_sin_gen::name_mul1, true},
        {cumsum_sin_gen::name_transpose2, true},
        {cumsum_sin_gen::name_mul2, true},
        {cumsum_sin_gen::name_interpolate, true},
        {cumsum_sin_gen::name_transpose3, true},
        {cumsum_sin_gen::name_sin, true},
    };
    cumsum_sin_gen::run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompCumSumSinGen_NoCumSumUpstream_NoOp) {
    auto model = cumsum_sin_gen::create_model_not_to_match();
    std::unordered_map<std::string, bool> expected_status = {
        {cumsum_sin_gen::name_mul1, false},
        {cumsum_sin_gen::name_sin, false},
    };
    cumsum_sin_gen::run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompCumSumSinGen_MissingSecondMultiply_NoOp) {
    auto model = cumsum_sin_gen::create_model_missing_mul2();
    // No second Multiply — pattern must not match.
    std::unordered_map<std::string, bool> expected_status = {
        {cumsum_sin_gen::name_cumsum, false},
        {cumsum_sin_gen::name_mul1, false},
        {cumsum_sin_gen::name_transpose2, false},
        {cumsum_sin_gen::name_interpolate, false},
        {cumsum_sin_gen::name_transpose3, false},
        {cumsum_sin_gen::name_sin, false},
    };
    cumsum_sin_gen::run_test(model, expected_status);
}

namespace hifigan_sin_gen {
const std::string name_multiply = "multiply";
const std::string name_interpolate = "interpolate";
const std::string name_transpose = "transpose";
const std::string name_sin = "sin";

// This model creates the exact pattern that DisableFP16CompSinGen is looking for.
// multiply - interpolate - transpose - sin
std::shared_ptr<ov::Model> create_model_to_match() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    // Pattern part 1: multiply
    auto mul_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {1.0f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(input, mul_const);
    multiply->set_friendly_name(name_multiply);

    // Pattern part 2: interpolate
    ov::op::v4::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v4::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ov::op::v4::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;

    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 64});
    auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {1.0f, 2.0f});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto interpolate = std::make_shared<ov::op::v4::Interpolate>(multiply, target_shape, scales, axes, attrs);
    interpolate->set_friendly_name(name_interpolate);

    // Pattern part 3: transpose
    auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(interpolate, order);
    transpose->set_friendly_name(name_transpose);

    // Pattern part 4: sin
    auto sin = std::make_shared<ov::op::v0::Sin>(transpose);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

// This model has a similar structure but doesn't match the specific pattern.
std::shared_ptr<ov::Model> create_model_not_to_match() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    auto mul_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {1.0f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(input, mul_const);
    multiply->set_friendly_name(name_multiply);

    auto sin = std::make_shared<ov::op::v0::Sin>(multiply);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

void run_test(std::shared_ptr<ov::Model> model,
              const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16ComSinGenPatternForHiFiGAN>();

    precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map);

    manager.run_passes(model);

    for (const auto& op : model->get_ops()) {
        auto it = expected_fp16_disabled_status.find(op->get_friendly_name());
        if (it != expected_fp16_disabled_status.end()) {
            bool expected_status = it->second;
            if (expected_status) {
                ASSERT_TRUE(ov::is_conversion_disabled(op, ov::element::f16))
                    << "FP16 compression is not disabled for node: " << op->get_friendly_name();
            } else {
                ASSERT_FALSE(ov::is_conversion_disabled(op, ov::element::f16))
                    << "FP16 compression is unexpectedly disabled for node: " << op->get_friendly_name();
            }
        }
    }
}
}  // namespace hifigan_sin_gen

TEST(TransformationTests, DisableFP16CompSinGen_Positive) {
    auto model = hifigan_sin_gen::create_model_to_match();
    // In the matching pattern, sin should have FP16 compression disabled.
    std::unordered_map<std::string, bool> expected_status = {
        {hifigan_sin_gen::name_multiply, true},
        {hifigan_sin_gen::name_interpolate, true},
        {hifigan_sin_gen::name_transpose, true},
        {hifigan_sin_gen::name_sin, true},
    };
    hifigan_sin_gen::run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompSinGen_Negative) {
    auto model = hifigan_sin_gen::create_model_not_to_match();
    // In the non-matching model, no node should have FP16 compression disabled by the pass.
    std::unordered_map<std::string, bool> expected_status = {
        {hifigan_sin_gen::name_multiply, false},
        {hifigan_sin_gen::name_sin, false},
    };
    hifigan_sin_gen::run_test(model, expected_status);
}

}  // namespace

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/l2_norm_to_rms.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/convert_precision.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

constexpr size_t hidden = 64;
constexpr float eps = 1e-6f;

// x / sqrt(ReduceSum(x^2, axis) + eps) * gamma, as an ONNX LpNormalization(p=2) followed by a scale exports it.
std::shared_ptr<ov::Node> l2_norm(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& gamma, int64_t axis = -1, bool keep_dims = true) {
    auto power = std::make_shared<ov::op::v1::Power>(x, ov::op::v0::Constant::create(x.get_element_type(), {}, {2.f}));
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    auto sum = std::make_shared<ov::op::v1::ReduceSum>(power, axes, keep_dims);
    auto add = std::make_shared<ov::op::v1::Add>(sum, ov::op::v0::Constant::create(x.get_element_type(), {}, {eps}));
    auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add);
    auto div = std::make_shared<ov::op::v1::Divide>(x, sqrt);
    return std::make_shared<ov::op::v1::Multiply>(div, gamma);
}

std::shared_ptr<ov::Node> rms(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& gamma, ov::element::Type out_type) {
    auto inv_sqrt_n = ov::op::v0::Constant::create(gamma.get_element_type(), {}, {1.0 / std::sqrt(static_cast<double>(hidden))});
    auto scaled_gamma = std::make_shared<ov::op::v1::Multiply>(gamma, inv_sqrt_n);
    return std::make_shared<ov::op::internal::RMS>(x, scaled_gamma, eps / static_cast<float>(hidden), out_type);
}

}  // namespace

TEST_F(TransformationTestsF, L2NormToRMS) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        auto norm = l2_norm(x, gamma);
        model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
        manager.register_pass<L2NormToRMS>(1024);
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rms(x, gamma, ov::element::f32)}, ov::ParameterVector{x, gamma});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// f16 activations normalized in f32 and converted back: RMS takes the f16 input and writes f16 directly.
TEST_F(TransformationTestsF, L2NormToRMSWithConverts) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        auto x_f32 = std::make_shared<ov::op::v0::Convert>(x, ov::element::f32);
        auto norm = l2_norm(x_f32, gamma);
        auto out = std::make_shared<ov::op::v0::Convert>(norm, ov::element::f16);
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x, gamma});
        manager.register_pass<L2NormToRMS>(1024);
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rms(x, gamma, ov::element::f16)}, ov::ParameterVector{x, gamma});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// Not normalized over the last axis: left alone.
TEST_F(TransformationTestsF, L2NormToRMSNotLastAxis) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, hidden, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden, 1});
    auto norm = l2_norm(x, gamma, 1);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

// Without keep_dims the reduced tensor does not broadcast against x the way RMS needs: left alone.
TEST_F(TransformationTestsF, L2NormToRMSNoKeepDims) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    auto norm = l2_norm(x, gamma, -1, false);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

// A row too long for one work group of the RMS kernel (hidden / 8 > max work group size): left alone.
TEST_F(TransformationTestsF, L2NormToRMSRowTooLong) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    auto norm = l2_norm(x, gamma);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(4);
}

// gamma {1, 1, n} still scales the normalized axis only: fused.
TEST_F(TransformationTestsF, L2NormToRMSGammaWithLeadingOnes) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, hidden});
        auto norm = l2_norm(x, gamma);
        model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
        manager.register_pass<L2NormToRMS>(1024);
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, hidden});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rms(x, gamma, ov::element::f32)}, ov::ParameterVector{x, gamma});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// gamma {n, 1} has n elements but scales each row, not the normalized axis: left alone.
TEST_F(TransformationTestsF, L2NormToRMSGammaColumn) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, hidden, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden, 1});
    auto norm = l2_norm(x, gamma);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

// gamma with more dims than x would change the output rank: left alone.
TEST_F(TransformationTestsF, L2NormToRMSGammaHigherRank) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, hidden});
    auto norm = l2_norm(x, gamma);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

// Rank 6 has no RMS implementation on GPU: left alone.
TEST_F(TransformationTestsF, L2NormToRMSRankSix) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 1, 1, 4, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    auto norm = l2_norm(x, gamma);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

namespace {

struct ResidualModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::Node> prev_residual;  // residual Add of the previous block
    std::shared_ptr<ov::Node> residual;       // residual Add feeding the normalization
    std::shared_ptr<ov::Node> bias_add;       // Add with a constant input on the branch
};

// (h + b) + (branch + bias) -> L2 norm * gamma: a residual stream feeding the normalization
ResidualModel residual_model() {
    const ov::Shape shape{1, 16, hidden};
    auto h = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto branch = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto prev_residual = std::make_shared<ov::op::v1::Add>(h, b);
    auto bias_add = std::make_shared<ov::op::v1::Add>(branch, ov::op::v0::Constant::create(ov::element::f32, {hidden}, {0.5f}));
    auto residual = std::make_shared<ov::op::v1::Add>(prev_residual, bias_add);
    auto gamma = ov::op::v0::Constant::create(ov::element::f32, {hidden}, {1.5f});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(residual, gamma)}, ov::ParameterVector{h, b, branch});
    return {model, prev_residual, residual, bias_add};
}

// As in the GPU pipeline: L2NormToRMS before ConvertPrecision, then the late L2NormToRMS run
void early_fuse_convert_precision_late_fuse(const std::shared_ptr<ov::Model>& model, bool skip_residual_input) {
    ov::pass::Manager manager;
    manager.register_pass<L2NormToRMS>(1024, skip_residual_input);
    manager.register_pass<ov::pass::ConvertPrecision>(::precisions_map{{ov::element::f32, ov::element::f16}},
                                                      ::type_to_fuse_map{},
                                                      true,    // keep_precision_sensitive_in_fp32
                                                      false);  // convert_input_output_precision
    manager.register_pass<L2NormToRMS>(1024);
    manager.run_passes(model);
}

std::shared_ptr<ov::Node> find_rms(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::internal::RMS>(op)) {
            return op;
        }
    }
    return nullptr;
}

}  // namespace

// Without skip_residual_input the early fusion removes the Divide MarkDivWithEps keys on, so the residual stream
// drops to f16.
TEST(L2NormToRMSSkipResidualInput, DisabledResidualStreamIsF16) {
    auto m = residual_model();
    early_fuse_convert_precision_late_fuse(m.model, false);
    ASSERT_NE(find_rms(m.model), nullptr);
    EXPECT_EQ(m.residual->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(m.prev_residual->get_output_element_type(0), ov::element::f16);
}

// A norm fed by a residual Add is not fused early: ConvertPrecision keeps the residual stream in FP32 as without the
// early run, and the late run fuses the norm.
TEST(L2NormToRMSSkipResidualInput, ResidualFedNormLeftForLateRun) {
    auto m = residual_model();
    ov::pass::Manager early;
    early.register_pass<L2NormToRMS>(1024, true);
    early.run_passes(m.model);
    EXPECT_EQ(find_rms(m.model), nullptr);

    auto m2 = residual_model();
    early_fuse_convert_precision_late_fuse(m2.model, true);
    ASSERT_NE(find_rms(m2.model), nullptr);
    EXPECT_EQ(m2.residual->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(m2.prev_residual->get_output_element_type(0), ov::element::f32);
}

// An Add with a constant input is a bias, not a residual connection: the norm it feeds is fused early.
TEST(L2NormToRMSSkipResidualInput, NormAfterBiasAddFusedEarly) {
    auto m = residual_model();
    auto gamma = ov::op::v0::Constant::create(ov::element::f32, {hidden}, {1.5f});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(m.bias_add, gamma)}, m.model->get_parameters());
    ov::pass::Manager manager;
    manager.register_pass<L2NormToRMS>(1024, true);
    manager.run_passes(model);
    EXPECT_NE(find_rms(model), nullptr);
}

// A norm fused early with gamma dequantized from u16 up to 65535: the dequantization stays FP32 (a u16 -> f16
// Convert would give inf) and the folded gamma is finite.
TEST(L2NormToRMSKeepGammaFP32, U16DequantizedGammaStaysFinite) {
    const ov::Shape shape{1, 16, hidden};
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    std::vector<uint16_t> q(hidden, 40000);
    q[0] = 65535;
    auto weight = ov::op::v0::Constant::create(ov::element::u16, {hidden}, q);
    auto weight_f32 = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f32);
    auto zp = std::make_shared<ov::op::v0::Convert>(ov::op::v0::Constant::create(ov::element::u16, {}, {127}), ov::element::f32);
    auto shifted = std::make_shared<ov::op::v1::Subtract>(weight_f32, zp);
    auto gamma = std::make_shared<ov::op::v1::Multiply>(shifted, ov::op::v0::Constant::create(ov::element::f32, {}, {0.000855f}));
    auto model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(x, gamma)}, ov::ParameterVector{x});

    early_fuse_convert_precision_late_fuse(model, true);

    const auto rms_node = find_rms(model);
    ASSERT_NE(rms_node, nullptr);
    EXPECT_EQ(weight_f32->get_output_element_type(0), ov::element::f32);
    const auto folded_gamma = ov::util::get_constant_from_source(rms_node->input_value(1));
    ASSERT_NE(folded_gamma, nullptr);
    for (const auto v : folded_gamma->cast_vector<float>()) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/l2_norm_to_rms.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_ops/rms.hpp"
#include "test_utils.h"
#include "transformations/common_optimizations/rms_fusion.hpp"

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

// Decomposed RMSNorm exported with Power(add_eps, 0.5) -> Power(..., -1.0)
// rewrites Power(0.5) to Sqrt and fuses to ov::op::internal::RMS.
TEST_F(TransformationTestsF, PowerHalfDecomposedRMSFusesToRMS) {
    constexpr size_t norm_hidden = 1024;
    constexpr float norm_eps = 1e-5F;
    std::vector<ov::bfloat16> gamma_vals(norm_hidden, ov::bfloat16(1.25F));
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::PartialShape{1, -1, norm_hidden});
        auto c2 = ov::op::v0::Constant::create(ov::element::bf16, {1}, {2.0F});
        auto sq = std::make_shared<ov::op::v1::Power>(x, c2);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(sq, axes, true);
        auto c_eps = ov::op::v0::Constant::create(ov::element::bf16, {1}, {norm_eps});
        auto add_eps = std::make_shared<ov::op::v1::Add>(mean, c_eps);
        auto c_half = ov::op::v0::Constant::create(ov::element::bf16, {1}, {0.5F});
        auto pow_half = std::make_shared<ov::op::v1::Power>(add_eps, c_half);
        auto c_neg1 = ov::op::v0::Constant::create(ov::element::bf16, {1}, {-1.0F});
        auto rsqrt = std::make_shared<ov::op::v1::Power>(pow_half, c_neg1);
        auto mul_x = std::make_shared<ov::op::v1::Multiply>(x, rsqrt);
        auto gamma = ov::op::v0::Constant::create(ov::element::bf16, {1, 1, norm_hidden}, gamma_vals);
        auto out = std::make_shared<ov::op::v1::Multiply>(mul_x, gamma);
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
        manager.register_pass<ConvertPowerHalfToSqrt>();
        manager.register_pass<ov::pass::RMSFusion>(false, true, true);
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::PartialShape{1, -1, norm_hidden});
        auto gamma = ov::op::v0::Constant::create(ov::element::bf16, {1, 1, norm_hidden}, gamma_vals);
        auto fused_rms = std::make_shared<ov::op::internal::RMS>(x, gamma, norm_eps, ov::element::bf16);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fused_rms}, ov::ParameterVector{x});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// ConvertPowerHalfToSqrt matches Constant(0.5) of shape {1, 1, 1} wrapped in Convert, while leaving
// rank-expanding Power(scalar_x, {1}) untouched.
TEST_F(TransformationTestsF, PowerHalfWithConvertAndBroadcastGuard) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{});
        auto c_half_3d = ov::op::v0::Constant::create(ov::element::f16, {1, 1, 1}, {0.5F});
        auto c_half_f32 = std::make_shared<ov::op::v0::Convert>(c_half_3d, ov::element::f32);
        auto pow_x = std::make_shared<ov::op::v1::Power>(x, c_half_f32);
        auto pow_s = std::make_shared<ov::op::v1::Power>(s, ov::op::v0::Constant::create(ov::element::f32, {1}, {0.5F}));
        model = std::make_shared<ov::Model>(ov::OutputVector{pow_x, pow_s}, ov::ParameterVector{x, s});
        manager.register_pass<ConvertPowerHalfToSqrt>();
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{});
        auto sqrt_x = std::make_shared<ov::op::v0::Sqrt>(x);
        auto pow_s = std::make_shared<ov::op::v1::Power>(s, ov::op::v0::Constant::create(ov::element::f32, {1}, {0.5F}));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sqrt_x, pow_s}, ov::ParameterVector{x, s});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// Scalar NaN exponent and Convert(0.5 -> i32) integral destination must not be rewritten to Sqrt.
TEST_F(TransformationTestsF, PowerHalfRejectsNaNAndIntegralConvert) {
    auto x_f32 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4});
    auto x_i32 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 4});
    auto nan_exp = ov::op::v0::Constant::create(ov::element::f32, {1}, {std::numeric_limits<float>::quiet_NaN()});
    auto pow_nan = std::make_shared<ov::op::v1::Power>(x_f32, nan_exp);
    auto half_to_i32 = std::make_shared<ov::op::v0::Convert>(ov::op::v0::Constant::create(ov::element::f32, {1}, {0.5F}), ov::element::i32);
    auto pow_i32 = std::make_shared<ov::op::v1::Power>(x_i32, half_to_i32);
    model = std::make_shared<ov::Model>(ov::OutputVector{pow_nan, pow_i32}, ov::ParameterVector{x_f32, x_i32});
    manager.register_pass<ConvertPowerHalfToSqrt>();
}

// End-to-end GPU compilation & execution check: 6D decomposed Power(0.5) RMSNorm stays decomposed because GPU RMS
// supports only up to 5D layouts (bfyx / bfzyx).
TEST(L2NormToRMSTest, power_half_rank_six_rms_stays_decomposed_on_gpu) {
    constexpr size_t norm_hidden = 64;
    constexpr float norm_eps = 1e-5F;

    std::vector<float> gamma_f32(norm_hidden, 1.25F);
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 2, 2, 4, norm_hidden});
    auto sq = std::make_shared<ov::op::v1::Power>(x, ov::op::v0::Constant::create(ov::element::f32, {1}, {2.0F}));
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {5});
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(sq, axes, true);
    auto add_eps = std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, {1}, {norm_eps}));
    auto pow_half = std::make_shared<ov::op::v1::Power>(add_eps, ov::op::v0::Constant::create(ov::element::f32, {1}, {0.5F}));
    auto rsqrt = std::make_shared<ov::op::v1::Power>(pow_half, ov::op::v0::Constant::create(ov::element::f32, {1}, {-1.0F}));
    auto mul_x = std::make_shared<ov::op::v1::Multiply>(x, rsqrt);
    auto gamma = ov::op::v0::Constant::create(ov::element::f32, {norm_hidden}, gamma_f32);
    auto out = std::make_shared<ov::op::v1::Multiply>(mul_x, gamma);
    auto ov_model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});

    ov::Core core;
    ov::CompiledModel compiled;
    ASSERT_NO_THROW(compiled = core.compile_model(ov_model, tests::get_engine_gpu_device_name(), ov::hint::inference_precision(ov::element::f32)));

    auto runtime_model = compiled.get_runtime_model();
    bool found_rms = false;
    for (const auto& op : runtime_model->get_ordered_ops()) {
        const auto& rt = op->get_rt_info();
        auto it = rt.find("layerType");
        if (it != rt.end() && it->second.is<std::string>()) {
            const auto& layer_type = it->second.as<std::string>();
            if (layer_type == "rms" || layer_type == "RMS") {
                found_rms = true;
            }
        }
    }
    EXPECT_FALSE(found_rms) << "Expected 6D RMSNorm to remain decomposed on GPU";

    auto req = compiled.create_infer_request();
    std::vector<float> input_vals(ov::shape_size(ov::Shape{2, 2, 2, 2, 4, norm_hidden}), 2.0F);
    req.set_input_tensor(0, ov::Tensor(ov::element::f32, {2, 2, 2, 2, 4, norm_hidden}, input_vals.data()));
    ASSERT_NO_THROW(req.infer());
    const float* actual = req.get_output_tensor(0).data<const float>();
    const float expected_val = 2.0F / std::sqrt(4.0F + norm_eps) * 1.25F;
    EXPECT_NEAR(actual[0], expected_val, 1e-3F);
}

// End-to-end GPU execution check: decomposed Power(0.5) RMSNorm with large finite inputs (400..1700, where
// x^2 > 65504 overflows FP16 if left unfused) fuses to RMS and executes in FP32 accumulation without Inf/NaN.
TEST(L2NormToRMSTest, power_half_decomposed_rms_fuses_to_rms) {
    constexpr size_t seq_len = 4;
    constexpr size_t norm_hidden = 1024;
    constexpr float norm_eps = 1e-5F;

    std::vector<float> gamma_f32(norm_hidden);
    std::vector<ov::bfloat16> gamma_bf16(norm_hidden);
    for (size_t i = 0; i < norm_hidden; ++i) {
        gamma_f32[i] = 0.75F + 0.5F * static_cast<float>(i % 7) / 6.0F;
        gamma_bf16[i] = ov::bfloat16(gamma_f32[i]);
    }

    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::PartialShape{1, -1, norm_hidden});
    auto sq = std::make_shared<ov::op::v1::Power>(x, ov::op::v0::Constant::create(ov::element::bf16, {1}, {2.0F}));
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(sq, axes, true);
    auto add_eps = std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::bf16, {1}, {norm_eps}));
    auto pow_half = std::make_shared<ov::op::v1::Power>(add_eps, ov::op::v0::Constant::create(ov::element::bf16, {1}, {0.5F}));
    auto rsqrt = std::make_shared<ov::op::v1::Power>(pow_half, ov::op::v0::Constant::create(ov::element::bf16, {1}, {-1.0F}));
    auto mul_x = std::make_shared<ov::op::v1::Multiply>(x, rsqrt);
    auto gamma = ov::op::v0::Constant::create(ov::element::bf16, {1, 1, norm_hidden}, gamma_bf16);
    auto mul_gamma = std::make_shared<ov::op::v1::Multiply>(mul_x, gamma);
    auto out_f32 = std::make_shared<ov::op::v0::Convert>(mul_gamma, ov::element::f32);
    auto ov_model = std::make_shared<ov::Model>(ov::OutputVector{out_f32}, ov::ParameterVector{x});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, tests::get_engine_gpu_device_name(), ov::hint::inference_precision(ov::element::f16));
    auto runtime_model = compiled.get_runtime_model();
    bool found_rms = false;
    bool found_reduce_mean = false;
    for (const auto& op : runtime_model->get_ordered_ops()) {
        const auto& rt = op->get_rt_info();
        auto it = rt.find("layerType");
        if (it != rt.end() && it->second.is<std::string>()) {
            const auto& layer_type = it->second.as<std::string>();
            if (layer_type == "rms" || layer_type == "RMS") {
                found_rms = true;
            } else if (layer_type == "ReduceMean" || layer_type == "Reduce") {
                found_reduce_mean = true;
            }
        }
    }
    EXPECT_TRUE(found_rms) << "Expected decomposed Power(0.5) RMSNorm to fuse into a GPU RMS primitive";
    EXPECT_FALSE(found_reduce_mean) << "Expected ReduceMean to be fused into RMS";

    std::vector<ov::bfloat16> in_bf16(seq_len * norm_hidden);
    std::vector<float> expected(seq_len * norm_hidden);
    const float row_scales[seq_len] = {408.0F, 900.0F, 1400.0F, 1700.0F};
    for (size_t s = 0; s < seq_len; ++s) {
        float sum_sq = 0.0F;
        for (size_t i = 0; i < norm_hidden; ++i) {
            const float sign = (i % 2 == 0) ? 1.0F : -1.0F;
            const float v = sign * row_scales[s] * (0.5F + 0.5F * static_cast<float>(i % 16) / 15.0F);
            in_bf16[s * norm_hidden + i] = ov::bfloat16(v);
            const float v_q = static_cast<float>(in_bf16[s * norm_hidden + i]);
            sum_sq += v_q * v_q;
        }
        const float inv_rms = 1.0F / std::sqrt(sum_sq / static_cast<float>(norm_hidden) + norm_eps);
        for (size_t i = 0; i < norm_hidden; ++i) {
            const float v_q = static_cast<float>(in_bf16[s * norm_hidden + i]);
            expected[s * norm_hidden + i] = v_q * inv_rms * static_cast<float>(gamma_bf16[i]);
        }
    }

    auto req = compiled.create_infer_request();
    req.set_input_tensor(0, ov::Tensor(ov::element::bf16, {1, seq_len, norm_hidden}, in_bf16.data()));
    req.infer();
    const float* actual = req.get_output_tensor(0).data<const float>();
    for (size_t idx = 0; idx < expected.size(); ++idx) {
        ASSERT_TRUE(std::isfinite(actual[idx])) << "Non-finite output at index " << idx;
        EXPECT_NEAR(actual[idx], expected[idx], 2e-2F) << "Mismatch at index " << idx;
    }
}

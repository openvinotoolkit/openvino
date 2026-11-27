// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_power_half_to_sqrt.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/common_optimizations/rms_fusion.hpp"

using namespace testing;
using namespace ov::intel_gpu;

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
    constexpr size_t hidden = 64;
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

// Scalar NaN exponent, Convert(0.5 -> i32) integral destination, FP8 exponent constant, and non-exact f64
// exponent (0.5 + 1e-12) must not be rewritten to Sqrt and must not throw during pattern matching.
TEST_F(TransformationTestsF, PowerHalfRejectsNaNIntegralFP8AndNearHalfF64) {
    auto x_f32 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4});
    auto x_i32 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 4});
    auto x_f64 = std::make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::PartialShape{1, 4});

    auto nan_exp = ov::op::v0::Constant::create(ov::element::f32, {1}, {std::numeric_limits<float>::quiet_NaN()});
    auto pow_nan = std::make_shared<ov::op::v1::Power>(x_f32, nan_exp);

    auto half_to_i32 = std::make_shared<ov::op::v0::Convert>(ov::op::v0::Constant::create(ov::element::f32, {1}, {0.5F}), ov::element::i32);
    auto pow_i32 = std::make_shared<ov::op::v1::Power>(x_i32, half_to_i32);

    auto fp8_half = ov::op::v0::Constant::create(ov::element::f8e4m3, {1}, {ov::float8_e4m3(0.5F)});
    auto fp8_to_f32 = std::make_shared<ov::op::v0::Convert>(fp8_half, ov::element::f32);
    auto pow_fp8 = std::make_shared<ov::op::v1::Power>(x_f32, fp8_to_f32);

    auto near_half_f64 = ov::op::v0::Constant::create(ov::element::f64, {1}, {0.5 + 1e-12});
    auto pow_f64 = std::make_shared<ov::op::v1::Power>(x_f64, near_half_f64);

    model = std::make_shared<ov::Model>(ov::OutputVector{pow_nan, pow_i32, pow_fp8, pow_f64}, ov::ParameterVector{x_f32, x_i32, x_f64});
    manager.register_pass<ConvertPowerHalfToSqrt>();
}

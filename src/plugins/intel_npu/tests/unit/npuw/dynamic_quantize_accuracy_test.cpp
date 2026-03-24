// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for DynamicQuantize decomposition accuracy.
// Builds quantize→dequantize roundtrip models for each decomposition variant,
// feeds random f32 data, and measures reconstruction error (L2 norm, max abs error).
//
// The dequantize chain matches the actual product code in create_dequant_nodes():
//   deq = (convert(q_i8, f32) - convert(zp_i8, f32)) * scale_f32
// where scale_f32 is output[1] of each decomposition variant.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/openvino.hpp"

namespace {

using namespace ov;

// ============================================================================
// Accuracy metrics
// ============================================================================
struct AccuracyMetrics {
    double l2_norm;
    double rel_l2_norm;
    double max_abs_error;
    double mean_abs_error;
};

AccuracyMetrics compute_metrics(const float* original, const float* reconstructed, size_t count) {
    double sum_sq_err = 0.0;
    double sum_abs_err = 0.0;
    double max_abs = 0.0;
    double sum_sq_orig = 0.0;

    for (size_t i = 0; i < count; ++i) {
        double diff = static_cast<double>(original[i]) - static_cast<double>(reconstructed[i]);
        sum_sq_err += diff * diff;
        double abs_diff = std::abs(diff);
        sum_abs_err += abs_diff;
        if (abs_diff > max_abs) max_abs = abs_diff;
        sum_sq_orig += static_cast<double>(original[i]) * static_cast<double>(original[i]);
    }

    AccuracyMetrics m;
    m.l2_norm = std::sqrt(sum_sq_err);
    m.rel_l2_norm = (sum_sq_orig > 0.0) ? m.l2_norm / std::sqrt(sum_sq_orig) : 0.0;
    m.max_abs_error = max_abs;
    m.mean_abs_error = (count > 0) ? sum_abs_err / static_cast<double>(count) : 0.0;
    return m;
}

// ============================================================================
// Decomposition variant 1 (handcrafted symmetric-style)
// Matches DecomposeDynamicQuantize in kv_cache_compressed.cpp
// ============================================================================
std::shared_ptr<Model> build_roundtrip_v1(const Shape& shape, size_t reduction_axis) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, shape);
    input->set_friendly_name("input");

    auto cst_inv127 = op::v0::Constant::create(element::f32, Shape{}, {1.0f / 127.0f});
    auto cst_zero = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    float quant_max = 127.0f;
    float quant_min = -127.0f;

    auto axis_const = op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(reduction_axis)});

    auto reduce_min = std::make_shared<op::v1::ReduceMin>(input, axis_const, true);
    auto reduce_max = std::make_shared<op::v1::ReduceMax>(input, axis_const, true);

    auto clamped_min = std::make_shared<op::v0::Clamp>(reduce_min, quant_min, quant_max);
    auto clamped_max = std::make_shared<op::v0::Clamp>(reduce_max, quant_min, quant_max);

    auto range = std::make_shared<op::v1::Subtract>(clamped_max, clamped_min);
    auto scale = std::make_shared<op::v1::Multiply>(range, cst_inv127);

    auto zp_float = std::make_shared<op::v1::Divide>(cst_zero, scale);
    auto zp_rounded = std::make_shared<op::v5::Round>(zp_float, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto zp_clamped = std::make_shared<op::v0::Clamp>(zp_rounded, quant_min, quant_max);
    auto zp_i8 = std::make_shared<op::v0::Convert>(zp_clamped, element::i8);

    auto normalized = std::make_shared<op::v1::Divide>(input, scale);
    auto rounded = std::make_shared<op::v5::Round>(normalized, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto with_zp = std::make_shared<op::v1::Add>(rounded, zp_clamped);
    auto quantized_clamped = std::make_shared<op::v0::Clamp>(with_zp, quant_min, quant_max);
    auto quantized_i8 = std::make_shared<op::v0::Convert>(quantized_clamped, element::i8);

    auto q_f32 = std::make_shared<op::v0::Convert>(quantized_i8, element::f32);
    auto zp_f32_deq = std::make_shared<op::v0::Convert>(zp_i8, element::f32);
    auto sub_zp = std::make_shared<op::v1::Subtract>(q_f32, zp_f32_deq);
    auto reconstructed = std::make_shared<op::v1::Multiply>(sub_zp, scale);

    auto result = std::make_shared<op::v0::Result>(reconstructed);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{input}, "roundtrip_v1");
}

// ============================================================================
// Decomposition variant 2 (ONNX DynamicQuantizeLinear style, [0, 255] → u8)
// Based on DecomposeDynamicQuantize2 in kv_cache_compressed.cpp
// ============================================================================
std::shared_ptr<Model> build_roundtrip_v2(const Shape& shape, size_t reduction_axis) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, shape);
    input->set_friendly_name("input");

    auto axis_const = op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(reduction_axis)});
    auto zero_const = op::v0::Constant::create(element::f32, Shape{1}, {0.0f});
    auto quant_range_min = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    auto quant_range_max = op::v0::Constant::create(element::f32, Shape{}, {255.0f});
    auto quant_range_span = std::make_shared<op::v1::Subtract>(quant_range_max, quant_range_min);

    auto reduce_min = std::make_shared<op::v1::ReduceMin>(input, axis_const, true);
    auto x_min = std::make_shared<op::v1::Minimum>(zero_const, reduce_min);

    auto reduce_max = std::make_shared<op::v1::ReduceMax>(input, axis_const, true);
    auto x_max = std::make_shared<op::v1::Maximum>(zero_const, reduce_max);

    auto x_span = std::make_shared<op::v1::Subtract>(x_max, x_min);
    auto y_scale = std::make_shared<op::v1::Divide>(x_span, quant_range_span);

    auto x_min_div_scale = std::make_shared<op::v1::Divide>(x_min, y_scale);
    auto intermediate_zp = std::make_shared<op::v5::Round>(
        std::make_shared<op::v1::Subtract>(quant_range_min, x_min_div_scale),
        op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto y_zp = std::make_shared<op::v0::Convert>(
        std::make_shared<op::v0::Clamp>(intermediate_zp, 0.0, 255.0), element::u8);

    auto x_scaled = std::make_shared<op::v1::Divide>(input, y_scale);
    auto x_rounded = std::make_shared<op::v5::Round>(x_scaled, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto y_zp_f32 = std::make_shared<op::v0::Convert>(y_zp, element::f32);
    auto result_shifted = std::make_shared<op::v1::Add>(x_rounded, y_zp_f32);
    auto result_clamped = std::make_shared<op::v0::Clamp>(result_shifted, 0.0, 255.0);
    auto quantized_u8 = std::make_shared<op::v0::Convert>(result_clamped, element::u8);

    auto q_f32 = std::make_shared<op::v0::Convert>(quantized_u8, element::f32);
    auto zp_f32_deq = std::make_shared<op::v0::Convert>(y_zp, element::f32);
    auto sub_zp = std::make_shared<op::v1::Subtract>(q_f32, zp_f32_deq);
    auto reconstructed = std::make_shared<op::v1::Multiply>(sub_zp, y_scale);

    auto result = std::make_shared<op::v0::Result>(reconstructed);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{input}, "roundtrip_v2");
}

// ============================================================================
// Decomposition variant 3 (compiler pattern style)
// Matches DecomposeDynamicQuantize3 in kv_cache_compressed.cpp
// ============================================================================
std::shared_ptr<Model> build_roundtrip_v3(const Shape& shape, size_t reduction_axis) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, shape);
    input->set_friendly_name("input");

    auto axis_const = op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(reduction_axis)});
    auto cst_255 = op::v0::Constant::create(element::f32, Shape{1}, {255.0f});
    auto cst_inv255 = op::v0::Constant::create(element::f32, Shape{1}, {1.0f / 255.0f});

    auto reduceMin = std::make_shared<op::v1::ReduceMin>(input, axis_const, true);
    auto reduceMax = std::make_shared<op::v1::ReduceMax>(input, axis_const, true);

    auto minClamped = std::make_shared<op::v0::Clamp>(reduceMin, std::numeric_limits<float>::lowest(), 0.0);
    auto maxClamped = std::make_shared<op::v0::Clamp>(reduceMax, 0.0, std::numeric_limits<float>::max());

    auto subtractSpan = std::make_shared<op::v1::Subtract>(maxClamped, minClamped);
    auto multiplyScale = std::make_shared<op::v1::Multiply>(subtractSpan, cst_inv255);

    auto multiplyInput = std::make_shared<op::v1::Multiply>(input, cst_255);
    auto divideSpan = std::make_shared<op::v1::Divide>(multiplyInput, subtractSpan);
    auto roundSpan = std::make_shared<op::v5::Round>(divideSpan, op::v5::Round::RoundMode::HALF_TO_EVEN);

    auto cst_qmin = op::v0::Constant::create(element::f32, Shape{1}, {-128.0f});
    auto minDivScale = std::make_shared<op::v1::Divide>(minClamped, multiplyScale);
    auto zpFloat = std::make_shared<op::v1::Subtract>(cst_qmin, minDivScale);
    auto roundZp = std::make_shared<op::v5::Round>(zpFloat, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto clampZp = std::make_shared<op::v0::Clamp>(roundZp, -128.0, 127.0);
    auto convertZp = std::make_shared<op::v0::Convert>(clampZp, element::i8);

    auto addQuant = std::make_shared<op::v1::Add>(roundSpan, clampZp);
    auto clampOutput = std::make_shared<op::v0::Clamp>(addQuant, -128.0, 127.0);
    auto quantized_i8 = std::make_shared<op::v0::Convert>(clampOutput, element::i8);

    auto q_f32 = std::make_shared<op::v0::Convert>(quantized_i8, element::f32);
    auto zp_f32_deq = std::make_shared<op::v0::Convert>(convertZp, element::f32);
    auto sub_zp = std::make_shared<op::v1::Subtract>(q_f32, zp_f32_deq);
    auto reconstructed = std::make_shared<op::v1::Multiply>(sub_zp, multiplyScale);

    auto result = std::make_shared<op::v0::Result>(reconstructed);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{input}, "roundtrip_v3");
}

// ============================================================================
// Data generation helpers
// ============================================================================
std::vector<float> generate_uniform_data(size_t count, float min_val, float max_val, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<float> data(count);
    for (auto& v : data) {
        v = dist(gen);
    }
    return data;
}

std::vector<float> generate_gen_gaussian_data(size_t count, float mu, float alpha, float beta, unsigned seed) {
    std::mt19937 gen(seed);
    std::gamma_distribution<double> gamma_dist(1.0 / static_cast<double>(beta), 1.0);
    std::bernoulli_distribution sign_dist(0.5);

    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        double g = gamma_dist(gen);
        double abs_x = static_cast<double>(alpha) * std::pow(g, 1.0 / static_cast<double>(beta));
        double sign = sign_dist(gen) ? 1.0 : -1.0;
        data[i] = static_cast<float>(static_cast<double>(mu) + sign * abs_x);
    }
    return data;
}

// ============================================================================
// Helper: evaluate a roundtrip model
// ============================================================================
AccuracyMetrics evaluate_roundtrip(const std::shared_ptr<Model>& model,
                                   const float* input_data,
                                   size_t element_count) {
    auto input_tensor = Tensor(element::f32, model->get_parameters()[0]->get_shape());
    std::memcpy(input_tensor.data<float>(), input_data, element_count * sizeof(float));

    auto output_tensor = Tensor(element::f32, model->get_results()[0]->get_shape());

    TensorVector inputs{input_tensor};
    TensorVector outputs{output_tensor};

    bool ok = model->evaluate(outputs, inputs);
    EXPECT_TRUE(ok) << "Model evaluation failed for " << model->get_friendly_name();

    return compute_metrics(input_data, outputs[0].data<float>(), element_count);
}

// ============================================================================
// Test parameter definition
// ============================================================================
enum class DistributionKind { Uniform, GenGaussian };

struct DQTestParams {
    std::string name;
    Shape shape;
    size_t reduction_axis;
    unsigned seed;
    DistributionKind dist_kind;
    // Uniform distribution params
    float uniform_min;
    float uniform_max;
    // Generalized Gaussian params
    float ggd_mu;
    float ggd_alpha;
    float ggd_beta;
    // Whether to run cross-variant comparison (max/min ratio check)
    bool cross_variant_check;
};

std::ostream& operator<<(std::ostream& os, const DQTestParams& p) {
    return os << p.name;
}

// ============================================================================
// Parameterized test fixture
// ============================================================================
class DynamicQuantizeAccuracyTest : public ::testing::TestWithParam<DQTestParams> {
protected:
    static constexpr double kRelL2Threshold = 0.02;

    static void print_metrics(const std::string& label, const AccuracyMetrics& m) {
        std::cout << "  " << label
                  << ": L2=" << m.l2_norm
                  << ", relL2=" << m.rel_l2_norm
                  << ", maxAbs=" << m.max_abs_error
                  << ", meanAbs=" << m.mean_abs_error
                  << std::endl;
    }

    std::vector<float> generate_data(const DQTestParams& p, size_t count) {
        switch (p.dist_kind) {
        case DistributionKind::Uniform:
            return generate_uniform_data(count, p.uniform_min, p.uniform_max, p.seed);
        case DistributionKind::GenGaussian:
            return generate_gen_gaussian_data(count, p.ggd_mu, p.ggd_alpha, p.ggd_beta, p.seed);
        default:
            return {};
        }
    }
};

TEST_P(DynamicQuantizeAccuracyTest, RoundtripAccuracy) {
    const auto& p = GetParam();

    size_t element_count = 1;
    for (auto d : p.shape) {
        element_count *= d;
    }

    auto data = generate_data(p, element_count);
    ASSERT_EQ(data.size(), element_count);

    auto model_v1 = build_roundtrip_v1(p.shape, p.reduction_axis);
    auto model_v2 = build_roundtrip_v2(p.shape, p.reduction_axis);
    auto model_v3 = build_roundtrip_v3(p.shape, p.reduction_axis);

    ASSERT_NO_THROW(model_v1->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v2->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v3->validate_nodes_and_infer_types());

    auto m1 = evaluate_roundtrip(model_v1, data.data(), element_count);
    auto m2 = evaluate_roundtrip(model_v2, data.data(), element_count);
    auto m3 = evaluate_roundtrip(model_v3, data.data(), element_count);

    std::cout << p.name << ":" << std::endl;
    print_metrics("V1 (handcrafted)", m1);
    print_metrics("V2 (ONNX-u8)", m2);
    print_metrics("V3 (compiler-pattern)", m3);

    EXPECT_LT(m1.rel_l2_norm, kRelL2Threshold) << "V1 relative L2 too high";
    EXPECT_LT(m2.rel_l2_norm, kRelL2Threshold) << "V2 relative L2 too high";
    EXPECT_LT(m3.rel_l2_norm, kRelL2Threshold) << "V3 relative L2 too high";

    if (p.cross_variant_check) {
        double max_rel = std::max({m1.rel_l2_norm, m2.rel_l2_norm, m3.rel_l2_norm});
        double min_rel = std::min({m1.rel_l2_norm, m2.rel_l2_norm, m3.rel_l2_norm});
        if (min_rel > 1e-9) {
            EXPECT_LT(max_rel / min_rel, 4.0) << "Variants differ too much in accuracy";
        }
    }
}

// ============================================================================
// Helper macros for concise param construction
// ============================================================================
// clang-format off
static DQTestParams make_uniform(const std::string& name, Shape shape, size_t axis,
                                 float lo, float hi, unsigned seed,
                                 bool cross_check = false) {
    return {name, shape, axis, seed, DistributionKind::Uniform,
            lo, hi, 0.0f, 0.0f, 0.0f, cross_check};
}

static DQTestParams make_ggd(const std::string& name, Shape shape, size_t axis,
                             float mu, float alpha, float beta, unsigned seed,
                             bool cross_check = false) {
    return {name, shape, axis, seed, DistributionKind::GenGaussian,
            0.0f, 0.0f, mu, alpha, beta, cross_check};
}
// clang-format on

// ============================================================================
// Test instantiation — Uniform distributions
// ============================================================================
static constexpr size_t kKeyAxis = 3;
static constexpr size_t kValueAxis = 2;

INSTANTIATE_TEST_SUITE_P(
    Uniform,
    DynamicQuantizeAccuracyTest,
    ::testing::Values(
        make_uniform("SmallRange_Key",            {1, 8, 64, 128},   kKeyAxis,   -1.0f,  1.0f,  42),
        make_uniform("SmallRange_Value",          {1, 8, 64, 128},   kValueAxis, -1.0f,  1.0f,  123),
        make_uniform("WiderRange_Key",            {1, 8, 64, 128},   kKeyAxis,   -10.0f, 10.0f, 77),
        make_uniform("AsymmetricPositive_Key",    {1, 8, 64, 128},   kKeyAxis,   0.0f,   5.0f,  999),
        make_uniform("CrossVariant",              {1, 8, 64, 128},   kKeyAxis,   -5.0f,  5.0f,  55, true),
        make_uniform("LargerSequence",            {1, 8, 1024, 128}, kKeyAxis,   -3.0f,  3.0f,  314)
    ),
    [](const ::testing::TestParamInfo<DQTestParams>& info) { return info.param.name; }
);

// ============================================================================
// Test instantiation — Generalized Gaussian (fitted from real KV-cache dumps)
// ============================================================================
// Parameters fitted from real KV-cache activations (key-cache):
//   Layer 25 (worst):  mu= 0.018, alpha=1.121, beta=0.923  (heaviest tails)
//   Layer 06 (mid):    mu= 0.032, alpha=1.388, beta=1.016  (near-Laplace)
//   Layer 15 (good):   mu= 0.039, alpha=1.616, beta=1.104
//   Layer 01 (best):   mu=-0.027, alpha=1.830, beta=1.271  (closest to normal)

INSTANTIATE_TEST_SUITE_P(
    GenGaussian,
    DynamicQuantizeAccuracyTest,
    ::testing::Values(
        make_ggd("HeavyTails_Key",         {1, 8, 1024, 128},   kKeyAxis, 0.018f, 1.121f, 0.923f, 42),
        make_ggd("NearLaplace_Key",        {1, 8, 1024, 128},   kKeyAxis, 0.032f, 1.388f, 1.016f, 77),
        make_ggd("NearNormal_Key",         {1, 8, 1024, 128},   kKeyAxis, -0.027f, 1.830f, 1.271f, 123),
        make_ggd("HeavyTails_Large",       {1, 8, 1024, 128}, kKeyAxis, 0.018f, 1.121f, 0.923f, 314),
        make_ggd("CrossVariant_GGD",       {1, 8, 1024, 128},   kKeyAxis, 0.039f, 1.616f, 1.104f, 55, true)
    ),
    [](const ::testing::TestParamInfo<DQTestParams>& info) { return info.param.name; }
);

}  // namespace

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
    double l2_norm;        // L2 (Euclidean) distance between original and reconstructed
    double rel_l2_norm;    // L2 distance normalized by input L2 norm
    double max_abs_error;  // Maximum absolute element-wise error
    double mean_abs_error; // Mean absolute element-wise error
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
//   scale = (clamp(max) - clamp(min)) * (1/127)
//   zp = round(clamp(0 / scale, -127, 127))
//   q = clamp(round(x / scale) + zp, -127, 127) → i8
//   output[1] = scale (as f32)
//   deq = (q_f32 - zp_f32) * scale
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
    auto scale = std::make_shared<op::v1::Multiply>(range, cst_inv127);  // output[1]

    auto zp_float = std::make_shared<op::v1::Divide>(cst_zero, scale);
    auto zp_rounded = std::make_shared<op::v5::Round>(zp_float, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto zp_clamped = std::make_shared<op::v0::Clamp>(zp_rounded, quant_min, quant_max);
    auto zp_i8 = std::make_shared<op::v0::Convert>(zp_clamped, element::i8);  // output[2]

    auto normalized = std::make_shared<op::v1::Divide>(input, scale);
    auto rounded = std::make_shared<op::v5::Round>(normalized, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto with_zp = std::make_shared<op::v1::Add>(rounded, zp_clamped);
    auto quantized_clamped = std::make_shared<op::v0::Clamp>(with_zp, quant_min, quant_max);
    auto quantized_i8 = std::make_shared<op::v0::Convert>(quantized_clamped, element::i8);  // output[0]

    // Dequantize: matches create_dequant_nodes() — (convert(q,f32) - convert(zp,f32)) * scale
    auto q_f32 = std::make_shared<op::v0::Convert>(quantized_i8, element::f32);
    auto zp_f32_deq = std::make_shared<op::v0::Convert>(zp_i8, element::f32);
    auto sub_zp = std::make_shared<op::v1::Subtract>(q_f32, zp_f32_deq);
    auto reconstructed = std::make_shared<op::v1::Multiply>(sub_zp, scale);

    auto result = std::make_shared<op::v0::Result>(reconstructed);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{input}, "roundtrip_v1");
}

// ============================================================================
// Decomposition variant 2 (ONNX DynamicQuantizeLinear style, [-128, 127])
// Matches DecomposeDynamicQuantize2 in kv_cache_compressed.cpp
// ============================================================================
//   x_min = min(0, ReduceMin(x)),  x_max = max(0, ReduceMax(x))
//   x_span = x_max - x_min,  quant_range_span = 255
//   y_scale = x_span / quant_range_span                   — output[1]
//   zp = clamp(round(qmin - (x_min / y_scale)), -128, 127) → i8   — output[2]
//   q = clamp(round(x * quant_range_span / x_span) + zp, -128, 127) → i8  — output[0]
//   deq = (q_f32 - zp_f32) * y_scale
std::shared_ptr<Model> build_roundtrip_v2(const Shape& shape, size_t reduction_axis) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, shape);
    input->set_friendly_name("input");

    auto axis_const = op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(reduction_axis)});
    auto zero_const = op::v0::Constant::create(element::f32, Shape{1}, {0.0f});
    auto quant_range_min = op::v0::Constant::create(element::f32, Shape{}, {-128.0f});
    auto quant_range_max = op::v0::Constant::create(element::f32, Shape{}, {127.0f});
    auto quant_range_span = std::make_shared<op::v1::Subtract>(quant_range_max, quant_range_min);  // 255


    auto reduce_min = std::make_shared<op::v1::ReduceMin>(input, axis_const, true);
    auto x_min = std::make_shared<op::v1::Minimum>(zero_const, reduce_min);

    auto reduce_max = std::make_shared<op::v1::ReduceMax>(input, axis_const, true);
    auto x_max = std::make_shared<op::v1::Maximum>(zero_const, reduce_max);

    auto x_span = std::make_shared<op::v1::Subtract>(x_max, x_min);
    auto y_scale = std::make_shared<op::v1::Divide>(x_span, quant_range_span);  // output[1]

    // ONNX spec: intermediate_zero_point = qmin - min(x) / y_scale
    // Note: this is qmin - (x_min / y_scale), NOT (qmin - x_min) / y_scale
    auto x_min_div_scale = std::make_shared<op::v1::Divide>(x_min, y_scale);
    auto intermediate_zp = std::make_shared<op::v5::Round>(
        std::make_shared<op::v1::Subtract>(quant_range_min, x_min_div_scale),
        op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto y_zp = std::make_shared<op::v0::Convert>(
        std::make_shared<op::v0::Clamp>(intermediate_zp, -128.0, 127.0), element::i8);  // output[2]

    // Quantize: exactly as source — round(x * quant_range_span / x_span) + zp
    auto x_scaled = std::make_shared<op::v1::Divide>(
        std::make_shared<op::v1::Multiply>(input, quant_range_span), x_span);
    auto x_rounded = std::make_shared<op::v5::Round>(x_scaled, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto y_zp_f32 = std::make_shared<op::v0::Convert>(y_zp, element::f32);
    auto result_shifted = std::make_shared<op::v1::Add>(x_rounded, y_zp_f32);
    auto result_clamped = std::make_shared<op::v0::Clamp>(result_shifted, -128.0, 127.0);
    auto quantized_i8 = std::make_shared<op::v0::Convert>(result_clamped, element::i8);  // output[0]

    // Dequantize: matches create_dequant_nodes() — (convert(q,f32) - convert(zp,f32)) * y_scale
    auto q_f32 = std::make_shared<op::v0::Convert>(quantized_i8, element::f32);
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
//   minClamped = clamp(ReduceMin, -inf, 0),  maxClamped = clamp(ReduceMax, 0, +inf)
//   span = maxClamped - minClamped
//   scale = span * (1/255)                                 — output[1] = multiplyScale
//   zp = clamp(round(qmin - (minClamped / scale)), -128, 127) → i8   — output[2]
//   q = clamp(round(x * 255 / span) + zp, -128, 127) → i8        — output[0]
//   deq = (q_f32 - zp_f32) * scale
std::shared_ptr<Model> build_roundtrip_v3(const Shape& shape, size_t reduction_axis) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, shape);
    input->set_friendly_name("input");

    auto axis_const = op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(reduction_axis)});
    auto cst_255 = op::v0::Constant::create(element::f32, Shape{1}, {255.0f});
    auto cst_inv255 = op::v0::Constant::create(element::f32, Shape{1}, {1.0f / 255.0f});
    auto cst_zero = op::v0::Constant::create(element::f32, Shape{1}, {0.0f});

    auto reduceMin = std::make_shared<op::v1::ReduceMin>(input, axis_const, true);
    auto reduceMax = std::make_shared<op::v1::ReduceMax>(input, axis_const, true);

    auto minClamped = std::make_shared<op::v0::Clamp>(reduceMin, std::numeric_limits<float>::lowest(), 0.0);
    auto maxClamped = std::make_shared<op::v0::Clamp>(reduceMax, 0.0, std::numeric_limits<float>::max());

    auto subtractSpan = std::make_shared<op::v1::Subtract>(maxClamped, minClamped);

    auto multiplyScale = std::make_shared<op::v1::Multiply>(subtractSpan, cst_inv255);  // output[1]

    // Quantize: exactly as source — round(x * 255 / span) + zp
    auto multiplyInput = std::make_shared<op::v1::Multiply>(input, cst_255);
    auto divideSpan = std::make_shared<op::v1::Divide>(multiplyInput, subtractSpan);
    auto roundSpan = std::make_shared<op::v5::Round>(divideSpan, op::v5::Round::RoundMode::HALF_TO_EVEN);

    // ONNX spec: zp = qmin - minClamped / scale
    // Note: this is qmin - (minClamped / scale), NOT (0 - minClamped) / scale
    auto cst_qmin = op::v0::Constant::create(element::f32, Shape{1}, {-128.0f});
    auto minDivScale = std::make_shared<op::v1::Divide>(minClamped, multiplyScale);
    auto zpFloat = std::make_shared<op::v1::Subtract>(cst_qmin, minDivScale);
    auto roundZp = std::make_shared<op::v5::Round>(zpFloat, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto clampZp = std::make_shared<op::v0::Clamp>(roundZp, -128.0, 127.0);
    auto convertZp = std::make_shared<op::v0::Convert>(clampZp, element::i8);  // output[2]

    auto addQuant = std::make_shared<op::v1::Add>(roundSpan, clampZp);
    auto clampOutput = std::make_shared<op::v0::Clamp>(addQuant, -128.0, 127.0);
    auto quantized_i8 = std::make_shared<op::v0::Convert>(clampOutput, element::i8);  // output[0]

    // Dequantize: matches create_dequant_nodes() — (convert(q,f32) - convert(zp,f32)) * scale
    auto q_f32 = std::make_shared<op::v0::Convert>(quantized_i8, element::f32);
    auto zp_f32_deq = std::make_shared<op::v0::Convert>(convertZp, element::f32);
    auto sub_zp = std::make_shared<op::v1::Subtract>(q_f32, zp_f32_deq);
    auto reconstructed = std::make_shared<op::v1::Multiply>(sub_zp, multiplyScale);

    auto result = std::make_shared<op::v0::Result>(reconstructed);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{input}, "roundtrip_v3");
}

// ============================================================================
// Helper: generate random f32 tensor data
// ============================================================================
std::vector<float> generate_random_data(size_t count, float min_val, float max_val, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<float> data(count);
    for (auto& v : data) {
        v = dist(gen);
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
// Test fixture
// ============================================================================
class DynamicQuantizeAccuracyTest : public ::testing::Test {
protected:
    static constexpr size_t kBatch = 1;
    static constexpr size_t kHeads = 8;
    static constexpr size_t kSeqLen = 64;
    static constexpr size_t kHeadDim = 128;

    Shape shape{kBatch, kHeads, kSeqLen, kHeadDim};
    size_t element_count = kBatch * kHeads * kSeqLen * kHeadDim;

    static constexpr size_t kKeyReductionAxis = 3;
    static constexpr size_t kValueReductionAxis = 2;

    // For int8 per-token quantization with 128-element tokens, theoretical best
    // relative error is ~1/(2*127) ≈ 0.4%. Allow up to 2% for rounding effects.
    static constexpr double kRelL2Threshold = 0.02;

    void print_metrics(const std::string& name, const AccuracyMetrics& m) {
        std::cout << "  " << name
                  << ": L2=" << m.l2_norm
                  << ", relL2=" << m.rel_l2_norm
                  << ", maxAbs=" << m.max_abs_error
                  << ", meanAbs=" << m.mean_abs_error
                  << std::endl;
    }
};

// ============================================================================
// Tests
// ============================================================================

TEST_F(DynamicQuantizeAccuracyTest, UniformSmallRange_KeyCache) {
    auto data = generate_random_data(element_count, -1.0f, 1.0f);

    auto model_v1 = build_roundtrip_v1(shape, kKeyReductionAxis);
    auto model_v2 = build_roundtrip_v2(shape, kKeyReductionAxis);
    auto model_v3 = build_roundtrip_v3(shape, kKeyReductionAxis);

    ASSERT_NO_THROW(model_v1->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v2->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v3->validate_nodes_and_infer_types());

    auto m1 = evaluate_roundtrip(model_v1, data.data(), element_count);
    auto m2 = evaluate_roundtrip(model_v2, data.data(), element_count);
    auto m3 = evaluate_roundtrip(model_v3, data.data(), element_count);

    std::cout << "UniformSmallRange_KeyCache:" << std::endl;
    print_metrics("V1 (handcrafted)", m1);
    print_metrics("V2 (ONNX-style)", m2);
    print_metrics("V3 (compiler-pattern)", m3);

    EXPECT_LT(m1.rel_l2_norm, kRelL2Threshold) << "V1 relative L2 too high";
    EXPECT_LT(m2.rel_l2_norm, kRelL2Threshold) << "V2 relative L2 too high";
    EXPECT_LT(m3.rel_l2_norm, kRelL2Threshold) << "V3 relative L2 too high";
}

TEST_F(DynamicQuantizeAccuracyTest, UniformSmallRange_ValueCache) {
    auto data = generate_random_data(element_count, -1.0f, 1.0f, 123);

    auto model_v1 = build_roundtrip_v1(shape, kValueReductionAxis);
    auto model_v2 = build_roundtrip_v2(shape, kValueReductionAxis);
    auto model_v3 = build_roundtrip_v3(shape, kValueReductionAxis);

    ASSERT_NO_THROW(model_v1->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v2->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v3->validate_nodes_and_infer_types());

    auto m1 = evaluate_roundtrip(model_v1, data.data(), element_count);
    auto m2 = evaluate_roundtrip(model_v2, data.data(), element_count);
    auto m3 = evaluate_roundtrip(model_v3, data.data(), element_count);

    std::cout << "UniformSmallRange_ValueCache:" << std::endl;
    print_metrics("V1 (handcrafted)", m1);
    print_metrics("V2 (ONNX-style)", m2);
    print_metrics("V3 (compiler-pattern)", m3);

    EXPECT_LT(m1.rel_l2_norm, kRelL2Threshold);
    EXPECT_LT(m2.rel_l2_norm, kRelL2Threshold);
    EXPECT_LT(m3.rel_l2_norm, kRelL2Threshold);
}

TEST_F(DynamicQuantizeAccuracyTest, WiderRange_KeyCache) {
    auto data = generate_random_data(element_count, -10.0f, 10.0f, 77);

    auto model_v1 = build_roundtrip_v1(shape, kKeyReductionAxis);
    auto model_v2 = build_roundtrip_v2(shape, kKeyReductionAxis);
    auto model_v3 = build_roundtrip_v3(shape, kKeyReductionAxis);

    ASSERT_NO_THROW(model_v1->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v2->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v3->validate_nodes_and_infer_types());

    auto m1 = evaluate_roundtrip(model_v1, data.data(), element_count);
    auto m2 = evaluate_roundtrip(model_v2, data.data(), element_count);
    auto m3 = evaluate_roundtrip(model_v3, data.data(), element_count);

    std::cout << "WiderRange_KeyCache:" << std::endl;
    print_metrics("V1 (handcrafted)", m1);
    print_metrics("V2 (ONNX-style)", m2);
    print_metrics("V3 (compiler-pattern)", m3);

    EXPECT_LT(m1.rel_l2_norm, kRelL2Threshold);
    EXPECT_LT(m2.rel_l2_norm, kRelL2Threshold);
    EXPECT_LT(m3.rel_l2_norm, kRelL2Threshold);
}

TEST_F(DynamicQuantizeAccuracyTest, AsymmetricPositiveRange) {
    auto data = generate_random_data(element_count, 0.0f, 5.0f, 999);

    auto model_v1 = build_roundtrip_v1(shape, kKeyReductionAxis);
    auto model_v2 = build_roundtrip_v2(shape, kKeyReductionAxis);
    auto model_v3 = build_roundtrip_v3(shape, kKeyReductionAxis);

    ASSERT_NO_THROW(model_v1->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v2->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v3->validate_nodes_and_infer_types());

    auto m1 = evaluate_roundtrip(model_v1, data.data(), element_count);
    auto m2 = evaluate_roundtrip(model_v2, data.data(), element_count);
    auto m3 = evaluate_roundtrip(model_v3, data.data(), element_count);

    std::cout << "AsymmetricPositiveRange:" << std::endl;
    print_metrics("V1 (handcrafted)", m1);
    print_metrics("V2 (ONNX-style)", m2);
    print_metrics("V3 (compiler-pattern)", m3);

    EXPECT_LT(m1.rel_l2_norm, kRelL2Threshold);
    EXPECT_LT(m2.rel_l2_norm, kRelL2Threshold);
    EXPECT_LT(m3.rel_l2_norm, kRelL2Threshold);
}

TEST_F(DynamicQuantizeAccuracyTest, CrossVariantComparison) {
    auto data = generate_random_data(element_count, -5.0f, 5.0f, 55);

    auto model_v1 = build_roundtrip_v1(shape, kKeyReductionAxis);
    auto model_v2 = build_roundtrip_v2(shape, kKeyReductionAxis);
    auto model_v3 = build_roundtrip_v3(shape, kKeyReductionAxis);

    ASSERT_NO_THROW(model_v1->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v2->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v3->validate_nodes_and_infer_types());

    auto m1 = evaluate_roundtrip(model_v1, data.data(), element_count);
    auto m2 = evaluate_roundtrip(model_v2, data.data(), element_count);
    auto m3 = evaluate_roundtrip(model_v3, data.data(), element_count);

    std::cout << "CrossVariantComparison [-5, 5]:" << std::endl;
    print_metrics("V1 (handcrafted)", m1);
    print_metrics("V2 (ONNX-style)", m2);
    print_metrics("V3 (compiler-pattern)", m3);

    EXPECT_LT(m1.rel_l2_norm, kRelL2Threshold);
    EXPECT_LT(m2.rel_l2_norm, kRelL2Threshold);
    EXPECT_LT(m3.rel_l2_norm, kRelL2Threshold);

    // No variant should be more than 4x worse than another
    double max_rel = std::max({m1.rel_l2_norm, m2.rel_l2_norm, m3.rel_l2_norm});
    double min_rel = std::min({m1.rel_l2_norm, m2.rel_l2_norm, m3.rel_l2_norm});
    if (min_rel > 1e-9) {
        EXPECT_LT(max_rel / min_rel, 4.0) << "Variants differ too much in accuracy";
    }
}

TEST_F(DynamicQuantizeAccuracyTest, LargerSequence) {
    Shape large_shape{1, 8, 1024, 128};
    size_t large_count = 1 * 8 * 1024 * 128;
    auto data = generate_random_data(large_count, -3.0f, 3.0f, 314);

    auto model_v1 = build_roundtrip_v1(large_shape, kKeyReductionAxis);
    auto model_v2 = build_roundtrip_v2(large_shape, kKeyReductionAxis);
    auto model_v3 = build_roundtrip_v3(large_shape, kKeyReductionAxis);

    ASSERT_NO_THROW(model_v1->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v2->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(model_v3->validate_nodes_and_infer_types());

    auto m1 = evaluate_roundtrip(model_v1, data.data(), large_count);
    auto m2 = evaluate_roundtrip(model_v2, data.data(), large_count);
    auto m3 = evaluate_roundtrip(model_v3, data.data(), large_count);

    std::cout << "LargerSequence [1,8,1024,128]:" << std::endl;
    print_metrics("V1 (handcrafted)", m1);
    print_metrics("V2 (ONNX-style)", m2);
    print_metrics("V3 (compiler-pattern)", m3);

    EXPECT_LT(m1.rel_l2_norm, kRelL2Threshold);
    EXPECT_LT(m2.rel_l2_norm, kRelL2Threshold);
    EXPECT_LT(m3.rel_l2_norm, kRelL2Threshold);
}

}  // namespace

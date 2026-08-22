// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "llm_compiled_model.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/openvino.hpp"

namespace {

constexpr std::size_t kHiddenSize = 512;
constexpr float kHiddenValue = 64.0f;
constexpr int8_t kWeightValue = 127;
constexpr int8_t kZeroPointValue = 126;
constexpr float kScaleValue = 0.00025f;

std::shared_ptr<ov::opset10::Constant> make_i8_constant(const ov::Shape& shape, int8_t value) {
    std::vector<int8_t> values(ov::shape_size(shape), value);
    return ov::opset10::Constant::create(ov::element::i8, shape, values);
}

std::shared_ptr<ov::opset10::Constant> make_f16_constant(const ov::Shape& shape, float value) {
    std::vector<ov::float16> values(ov::shape_size(shape), ov::float16(value));
    return ov::opset10::Constant::create(ov::element::f16, shape, values);
}

std::shared_ptr<ov::opset10::Parameter> make_hidden_parameter() {
    return std::make_shared<ov::opset10::Parameter>(ov::element::f16, ov::Shape{1, kHiddenSize});
}

std::shared_ptr<ov::Model> make_original_lm_head_model() {
    auto hidden = make_hidden_parameter();
    auto weights = make_i8_constant(ov::Shape{1, kHiddenSize}, kWeightValue);
    auto zero_point = make_i8_constant(ov::Shape{1, 1}, kZeroPointValue);
    auto scale = make_f16_constant(ov::Shape{1, 1}, kScaleValue);

    auto weights_f16 = std::make_shared<ov::opset10::Convert>(weights, ov::element::f16);
    auto zero_point_f16 = std::make_shared<ov::opset10::Convert>(zero_point, ov::element::f16);
    auto shifted_weights = std::make_shared<ov::opset10::Subtract>(weights_f16, zero_point_f16);
    auto scaled_weights = std::make_shared<ov::opset10::Multiply>(shifted_weights, scale);
    auto converted_weights = std::make_shared<ov::opset10::Convert>(scaled_weights, ov::element::f16);
    auto logits = std::make_shared<ov::opset10::MatMul>(hidden, converted_weights, false, true);

    return std::make_shared<ov::Model>(ov::OutputVector{logits}, ov::ParameterVector{hidden});
}

std::shared_ptr<ov::Model> make_reference_zp_decomposition_lm_head_model() {
    auto hidden = make_hidden_parameter();
    auto weights = make_i8_constant(ov::Shape{1, kHiddenSize}, kWeightValue);
    auto zero_point = make_i8_constant(ov::Shape{1, 1}, kZeroPointValue);
    auto scale = make_f16_constant(ov::Shape{1, 1}, kScaleValue);

    auto weights_f16 = std::make_shared<ov::opset10::Convert>(weights, ov::element::f16);
    auto scaled_weights = std::make_shared<ov::opset10::Multiply>(weights_f16, scale);
    auto raw_logits = std::make_shared<ov::opset10::MatMul>(hidden, scaled_weights, false, true);

    auto reduce_axis = ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto hidden_sum = std::make_shared<ov::opset10::ReduceSum>(hidden, reduce_axis, true);
    auto zero_point_f16 = std::make_shared<ov::opset10::Convert>(zero_point, ov::element::f16);
    auto scaled_zero_point = std::make_shared<ov::opset10::Multiply>(zero_point_f16, scale);
    auto correction = std::make_shared<ov::opset10::Multiply>(hidden_sum, scaled_zero_point);
    auto logits = std::make_shared<ov::opset10::Subtract>(raw_logits, correction);

    return std::make_shared<ov::Model>(ov::OutputVector{logits}, ov::ParameterVector{hidden});
}

std::shared_ptr<ov::Model> make_npuw_style_lm_head_model() {
    auto model = make_original_lm_head_model();
    OPENVINO_ASSERT(ov::npuw::apply_matmul_first_vocab(model));
    return model;
}

float evaluate_scalar(const std::shared_ptr<ov::Model>& model) {
    ov::Tensor hidden(ov::element::f16, ov::Shape{1, kHiddenSize});
    std::fill_n(hidden.data<ov::float16>(), hidden.get_size(), ov::float16(kHiddenValue));

    ov::Tensor output(ov::element::f16, ov::Shape{1, 1});
    const ov::TensorVector inputs{hidden};
    ov::TensorVector outputs{output};
    OPENVINO_ASSERT(model->evaluate(outputs, inputs));
    return static_cast<float>(outputs.front().data<const ov::float16>()[0]);
}

}  // namespace

TEST(MatMulFirstVocabAccuracyTest, NpuwPassMatchesReferenceZpDecomposition) {
    const auto original = evaluate_scalar(make_original_lm_head_model());
    const auto reference_zp_decomposition = evaluate_scalar(make_reference_zp_decomposition_lm_head_model());
    const auto npuw_style = evaluate_scalar(make_npuw_style_lm_head_model());

    ASSERT_TRUE(std::isfinite(original));
    EXPECT_TRUE(std::isfinite(reference_zp_decomposition));
    EXPECT_NEAR(original, reference_zp_decomposition, 32.0f);
    EXPECT_TRUE(std::isfinite(npuw_style));
    EXPECT_NEAR(original, npuw_style, 32.0f);
}

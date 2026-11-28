// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <variant>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/openvino.hpp"
#include "pyramid_attention.hpp"

namespace {

std::shared_ptr<ov::Model> build_left_aligned_quantized_attention_model() {
    using namespace ov;

    constexpr size_t num_heads = 4u;
    constexpr size_t head_dim = 16u;
    constexpr size_t past_len = 63u;
    constexpr size_t query_len = 1u;

    auto make_param = [](const std::string& name, const element::Type& element_type, const Shape& shape) {
        auto param = std::make_shared<op::v0::Parameter>(element_type, shape);
        param->set_friendly_name(name);
        param->output(0).get_tensor().set_names({name});
        return param;
    };

    auto query = make_param("query", element::f32, Shape{1, num_heads, query_len, head_dim});
    auto past_key = make_param("past_key_values.0.key", element::i8, Shape{1, past_len, num_heads, head_dim});
    auto present_key = make_param("present_key", element::i8, Shape{1, query_len, num_heads, head_dim});
    auto past_value = make_param("past_key_values.0.value", element::i8, Shape{1, num_heads, head_dim, past_len});
    auto present_value = make_param("present_value", element::i8, Shape{1, num_heads, head_dim, query_len});
    auto mask = make_param("attention_mask", element::f32, Shape{1, 1, query_len, past_len + query_len});

    auto key_concat = std::make_shared<op::v0::Concat>(OutputVector{present_key, past_key}, 1);
    auto key_transpose_order = op::v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});
    auto key_convert = std::make_shared<op::v0::Convert>(key_concat, element::f32);
    auto key = std::make_shared<op::v1::Transpose>(key_convert, key_transpose_order);

    auto value_concat = std::make_shared<op::v0::Concat>(OutputVector{present_value, past_value}, 3);
    auto value_transpose_order = op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2});
    auto value_convert = std::make_shared<op::v0::Convert>(value_concat, element::f32);
    auto value = std::make_shared<op::v1::Transpose>(value_convert, value_transpose_order);

    auto qk = std::make_shared<op::v0::MatMul>(query, key, false, true);
    auto masked_qk = std::make_shared<op::v1::Add>(qk, mask);
    auto softmax = std::make_shared<op::v8::Softmax>(masked_qk, 3);
    auto output = std::make_shared<op::v0::MatMul>(softmax, value);

    auto model = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(output)},
                                         ParameterVector{query, past_key, present_key, past_value, present_value, mask},
                                         "left_aligned_quantized_attention_model");
    model->validate_nodes_and_infer_types();
    return model;
}

TEST(PyramidAttentionLeftAlignmentTest, DetectsQuantizedPresentBeforePastLayout) {
    auto model = build_left_aligned_quantized_attention_model();

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);

    ASSERT_TRUE(result.has_value());
    const auto& contiguous = std::get<ov::npuw::function::PyramidValidationContiguousResult>(*result);
    EXPECT_TRUE(contiguous.data_left_aligned);
    EXPECT_EQ(contiguous.past_key_sequence_dims.at("past_key_values.0.key"), 1u);
    EXPECT_EQ(contiguous.past_value_sequence_dims.at("past_key_values.0.value"), 3u);
}

}  // namespace
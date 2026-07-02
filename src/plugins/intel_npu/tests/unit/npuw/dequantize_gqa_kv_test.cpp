// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/dequantize_gqa_kv.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"

namespace {

template <class Op>
std::size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ops();
    return std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<Op>(op);
    });
}

std::shared_ptr<ov::op::internal::GroupQueryAttention> find_gqa(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        if (auto gqa = ov::as_type_ptr<ov::op::internal::GroupQueryAttention>(op)) {
            return gqa;
        }
    }
    return nullptr;
}

// Build a GroupQueryAttention with an int8 PER_CHANNEL quantized KV cache: separate float Q/K/V, int8
// past_key/past_value, and f32 per-channel k_scale/v_scale at inputs 12/13 (com.microsoft layout). Mirrors
// the orca int8 op contract (num_heads=4, kv_num_heads=2, head_size=16 here for a small test).
std::shared_ptr<ov::Model> build_int8_gqa_model() {
    constexpr int64_t num_heads = 4, kv_num_heads = 2, head_size = 16, seq = 1, past = 8;
    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, num_heads, seq, head_size});
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, kv_num_heads, seq, head_size});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, kv_num_heads, seq, head_size});
    auto past_key =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{1, kv_num_heads, past, head_size});
    auto past_value =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{1, kv_num_heads, past, head_size});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});

    const auto scale_len = static_cast<size_t>(kv_num_heads * head_size);
    auto k_scale =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{scale_len}, std::vector<float>(scale_len, 0.02f));
    auto v_scale =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{scale_len}, std::vector<float>(scale_len, 0.03f));
    auto null = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});  // stand-in for optional slots

    // inputs: 0..6 mandatory, 7/8 rotary (unused here), 9/10/11 optional, 12/13 scales.
    ov::OutputVector args{query,
                          key,
                          value,
                          past_key,
                          past_value,
                          seqlens_k,
                          total_sequence_length,
                          null,
                          null,
                          null,
                          null,
                          null,
                          k_scale,
                          v_scale};
    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(args,
                                                                       num_heads,
                                                                       kv_num_heads,
                                                                       0.0f,   // scale (0 => default 1/sqrt(head))
                                                                       false,  // do_rotary
                                                                       false,  // rotary_interleaved
                                                                       8,      // kv_cache_bit_width
                                                                       "PER_CHANNEL",
                                                                       "PER_CHANNEL");

    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(gqa->output(0)),
                                std::make_shared<ov::op::v0::Result>(gqa->output(1)),
                                std::make_shared<ov::op::v0::Result>(gqa->output(2))};
    ov::ParameterVector params = {query, key, value, past_key, past_value, seqlens_k, total_sequence_length};
    return std::make_shared<ov::Model>(results, params, "int8_gqa_model");
}

// Same head config but a plain FLOAT KV cache (no quant) — the pass must leave it untouched.
std::shared_ptr<ov::Model> build_float_gqa_model() {
    constexpr int64_t num_heads = 4, kv_num_heads = 2, head_size = 16, seq = 1, past = 8;
    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, num_heads, seq, head_size});
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, kv_num_heads, seq, head_size});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, kv_num_heads, seq, head_size});
    auto past_key =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, kv_num_heads, past, head_size});
    auto past_value =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, kv_num_heads, past, head_size});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});

    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
        ov::OutputVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length},
        num_heads,
        kv_num_heads,
        0.0f,
        false,
        false);

    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(gqa->output(0)),
                                std::make_shared<ov::op::v0::Result>(gqa->output(1)),
                                std::make_shared<ov::op::v0::Result>(gqa->output(2))};
    ov::ParameterVector params = {query, key, value, past_key, past_value, seqlens_k, total_sequence_length};
    return std::make_shared<ov::Model>(results, params, "float_gqa_model");
}

}  // namespace

// The int8 quantized op is rewritten to a FLOAT GroupQueryAttention (no quant metadata, scale inputs dropped)
// sandwiched by dequant (Convert+Multiply on the KV inputs) and requant (Round+Clamp+Convert on present KV).
TEST(DequantizeGQAKVCache, RewritesQuantizedOpToFloatWithDequantRequant) {
    auto model = build_int8_gqa_model();
    ASSERT_NE(find_gqa(model), nullptr);
    ASSERT_TRUE(find_gqa(model)->is_kv_quantized());

    ov::npuw::DequantizeGQAKVCache pass;
    const bool changed = pass.run_on_model(model);
    EXPECT_TRUE(changed);

    // The op survives but is now float: quant metadata cleared and the scale inputs (12/13) dropped. (Trailing
    // real NullNode placeholders are also stripped for the orca FE graph; here the optional slots are stand-in
    // Constants, so only the scale-drop is asserted — the input count is < the original 14, never including 12/13.)
    auto gqa = find_gqa(model);
    ASSERT_NE(gqa, nullptr);
    EXPECT_FALSE(gqa->is_kv_quantized());
    EXPECT_EQ(gqa->get_kv_cache_bit_width(), 0);
    EXPECT_LT(gqa->get_input_size(), 14u);
    EXPECT_GE(gqa->get_input_size(), 7u);

    // The dequantized KV inputs (3/4) are now float, matching Q.
    EXPECT_EQ(gqa->get_input_element_type(3), gqa->get_input_element_type(0));
    EXPECT_EQ(gqa->get_input_element_type(4), gqa->get_input_element_type(0));

    // Requant restores int8 present KV at the model outputs.
    EXPECT_EQ(model->get_results()[1]->get_input_element_type(0), ov::element::i8);
    EXPECT_EQ(model->get_results()[2]->get_input_element_type(0), ov::element::i8);

    // Dequant/requant primitives are present (Round + Clamp are unique to the requant path).
    EXPECT_EQ(count_ops<ov::op::v5::Round>(model), 2u);  // K + V present requant
    EXPECT_EQ(count_ops<ov::op::v0::Clamp>(model), 2u);
    EXPECT_GT(count_ops<ov::op::v1::Multiply>(model), 0u);
}

// A float GQA (is_kv_quantized()==false) must be left completely untouched (no regression to the fp16 path).
TEST(DequantizeGQAKVCache, LeavesFloatOpUntouched) {
    auto model = build_float_gqa_model();
    ASSERT_NE(find_gqa(model), nullptr);
    ASSERT_FALSE(find_gqa(model)->is_kv_quantized());

    ov::npuw::DequantizeGQAKVCache pass;
    const bool changed = pass.run_on_model(model);

    EXPECT_FALSE(changed);
    EXPECT_EQ(count_ops<ov::op::v5::Round>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::Clamp>(model), 0u);
    auto gqa = find_gqa(model);
    ASSERT_NE(gqa, nullptr);
    EXPECT_EQ(gqa->get_input_size(), 7u);
}

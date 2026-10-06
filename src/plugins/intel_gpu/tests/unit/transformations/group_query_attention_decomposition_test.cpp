// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/group_query_attention_decomposition.hpp"

#include <gtest/gtest.h>

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"

namespace ov::test::intel_gpu {
namespace {

using QuantType = ov::op::internal::GroupQueryAttentionQuantType;

constexpr int64_t num_heads = 2;
constexpr int64_t kv_num_heads = 1;
constexpr int64_t head_size = 16;

// Positional order of the trailing ctor arguments, confirmed empirically:
// passing true into slot 10 raised
//     "sliding_window_cache requires local_window_size >= 1, got -1"
// which pins slot 10 to sliding_window_cache, and therefore slot 11 to
// smooth_softmax. Slot 12 is causal (exercised by the control test).
struct GQAConfig {
    float scale = 0.0f;   // 0.0f == "use 1/sqrt(head_size)"
    bool flag_a = false;  // do_rotary
    bool flag_b = false;  // rotary_interleaved
    int64_t kv_cache_bit_width = 0;
    QuantType kv_quant = QuantType::NONE;
    QuantType out_quant = QuantType::NONE;
    ov::element::Type cache_type = ov::element::i8;
    int64_t local_window_size = -1;  // >= 1 enables sliding window attention
    bool sliding_window_cache = false;
    bool smooth_softmax = false;  // adds an extra logit -> sink branch
    bool attention_bias = false;
    bool head_sink = false;
    bool causal = true;
    ov::PartialShape key_scale_shape{1};
    ov::PartialShape value_scale_shape{1};
    ov::Dimension past_len = ov::Dimension::dynamic();  // static == full-length static KV cache
};

std::shared_ptr<ov::Model> make_gqa_model(const GQAConfig& cfg) {
    const auto f32 = ov::element::f32;
    const auto past_len = cfg.past_len;

    auto query = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, num_heads, 1, head_size});
    auto key = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, 1, head_size});
    auto value = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, 1, head_size});
    const auto cache_type = cfg.kv_cache_bit_width ? cfg.cache_type : f32;
    const auto cache_head_size = cfg.kv_cache_bit_width == 4 ? head_size / 2 : head_size;
    auto past_key = std::make_shared<ov::op::v0::Parameter>(cache_type, ov::PartialShape{1, kv_num_heads, past_len, cache_head_size});
    auto past_value = std::make_shared<ov::op::v0::Parameter>(cache_type, ov::PartialShape{1, kv_num_heads, past_len, cache_head_size});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});

    ov::OutputVector inputs(14);
    inputs[0] = query;
    inputs[1] = key;
    inputs[2] = value;
    inputs[3] = past_key;
    inputs[4] = past_value;
    inputs[5] = seqlens_k;
    inputs[6] = total_sequence_length;
    for (size_t i = 7; i <= 13; ++i) {
        inputs[i] = ov::op::v0::Constant::create(f32, ov::Shape{0}, {});
    }
    ov::ParameterVector parameters{query, key, value, past_key, past_value, seqlens_k, total_sequence_length};

    if (cfg.attention_bias) {
        auto attention_bias = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, 1, 1, past_len});
        inputs[10] = attention_bias;
        parameters.push_back(attention_bias);
    }
    if (cfg.head_sink) {
        auto head_sink = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{num_heads});
        inputs[11] = head_sink;
        parameters.push_back(head_sink);
    }
    if (cfg.kv_cache_bit_width) {
        auto key_scale = std::make_shared<ov::op::v0::Parameter>(f32, cfg.key_scale_shape);
        auto value_scale = std::make_shared<ov::op::v0::Parameter>(f32, cfg.value_scale_shape);
        inputs[12] = key_scale;
        inputs[13] = value_scale;
        parameters.push_back(key_scale);
        parameters.push_back(value_scale);
    }

    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(inputs,
                                                                       num_heads,
                                                                       kv_num_heads,
                                                                       cfg.scale,
                                                                       cfg.flag_a,
                                                                       cfg.flag_b,
                                                                       cfg.kv_cache_bit_width,
                                                                       cfg.kv_quant,
                                                                       cfg.out_quant,
                                                                       cfg.local_window_size,
                                                                       cfg.sliding_window_cache,
                                                                       cfg.smooth_softmax,
                                                                       cfg.causal);

    ov::ResultVector results;
    for (const auto& output : gqa->outputs()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(output));
    }
    return std::make_shared<ov::Model>(results, parameters);
}

std::shared_ptr<ov::intel_gpu::op::SDPA> decompose_and_get_sdpa(const GQAConfig& cfg) {
    auto model = make_gqa_model(cfg);
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::GroupQueryAttentionDecomposition>();
    manager.run_passes(model);

    std::shared_ptr<ov::intel_gpu::op::SDPA> result;
    for (const auto& node : model->get_ordered_ops()) {
        EXPECT_FALSE(ov::is_type<ov::op::internal::GroupQueryAttention>(node));
        if (auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SDPA>(node)) {
            result = sdpa;
        }
    }
    return result;
}

// A mask produced by make_attention_mask() is a rank-4 broadcastable tensor.
// A rank-0 value in that slot means the optional inputs have shifted and the
// scale scalar landed there instead.
::testing::AssertionResult slot_holds_a_mask(const ov::Output<ov::Node>& slot) {
    const auto& ps = slot.get_partial_shape();
    if (ps.rank().is_static() && ps.rank().get_length() == 0) {
        return ::testing::AssertionFailure() << "attn_mask slot holds a rank-0 (scalar) value, produced by " << slot.get_node()->get_type_name() << " '"
                                             << slot.get_node()->get_friendly_name() << "' -- this is the scale scalar, not an attention mask";
    }
    return ::testing::AssertionSuccess();
}

void expect_fixed_int4_zero_point(const ov::Output<ov::Node>& zero_point) {
    EXPECT_EQ(zero_point.get_element_type(), ov::element::f16);
    EXPECT_EQ(zero_point.get_partial_shape(), ov::PartialShape({1, kv_num_heads, 1, head_size}));
    const auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(zero_point.get_node_shared_ptr());
    ASSERT_NE(broadcast, nullptr);
    const auto value = ov::as_type_ptr<ov::op::v0::Constant>(broadcast->input_value(0).get_node_shared_ptr());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->cast_vector<float>(), std::vector<float>{8.0f});
}

// Verify that GQAConfig maps sliding_window_cache and smooth_softmax
// to the intended constructor arguments. Re-checks this at runtime so a future ctor change
// cannot silently invalidate the rest of the suite.
TEST(GQADecompositionTest, maps_control_fields) {
    {
        GQAConfig cfg;
        cfg.sliding_window_cache = true;
        cfg.local_window_size = -1;
        EXPECT_ANY_THROW(make_gqa_model(cfg)) << "slot 10 is not sliding_window_cache -- GQAConfig field order is wrong";
    }
    {
        GQAConfig cfg;
        cfg.smooth_softmax = true;
        cfg.local_window_size = -1;
        EXPECT_NO_THROW(make_gqa_model(cfg)) << "slot 11 rejected local_window_size=-1, so it is not smooth_softmax "
                                                "-- GQAConfig field order is wrong";
    }
}

// Plain causal attention uses lower-right masking without an explicit mask.
TEST(GQADecompositionTest, causal_uses_lower_right_without_mask) {
    GQAConfig cfg;
    const auto sdpa = decompose_and_get_sdpa(cfg);

    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_size(), 3u) << "Q, K, V only -- the mask is elided here by design";
    EXPECT_TRUE(sdpa->get_causal());
    EXPECT_EQ(sdpa->get_causal_mask_alignment(), ov::intel_gpu::op::SDPA::CausalMaskAlignment::LOWER_RIGHT);
}

TEST(GQADecompositionTest, per_tensor_kv_uses_decompressed_sdpa) {
    GQAConfig cfg;
    cfg.kv_cache_bit_width = 8;
    cfg.kv_quant = QuantType::PER_TENSOR;
    cfg.out_quant = QuantType::PER_TENSOR;

    const auto sdpa = decompose_and_get_sdpa(cfg);

    ASSERT_NE(sdpa, nullptr);
    EXPECT_FALSE(sdpa->get_kv_compressed());
    ASSERT_EQ(sdpa->get_input_size(), 3u) << "Q, dequantized K, dequantized V";
    EXPECT_EQ(sdpa->input_value(1).get_element_type(), ov::element::f32);
    EXPECT_EQ(sdpa->input_value(2).get_element_type(), ov::element::f32);
}

TEST(GQADecompositionTest, per_channel_kv_uses_compressed_sdpa) {
    GQAConfig cfg;
    cfg.kv_cache_bit_width = 8;
    cfg.kv_quant = QuantType::PER_CHANNEL;
    cfg.out_quant = QuantType::PER_CHANNEL;
    cfg.key_scale_shape = ov::PartialShape{kv_num_heads * head_size};
    cfg.value_scale_shape = ov::PartialShape{kv_num_heads * head_size};

    const auto sdpa = decompose_and_get_sdpa(cfg);

    ASSERT_NE(sdpa, nullptr);
    ASSERT_TRUE(sdpa->get_kv_compressed());
    ASSERT_EQ(sdpa->get_input_size(), 5u) << "Q, K, V, K scale, V scale";
    EXPECT_EQ(sdpa->get_quantization_attrs().quantization_dt, ov::element::i8);
    EXPECT_EQ(sdpa->get_quantization_attrs().scale_dt, ov::element::f16);
    EXPECT_EQ(sdpa->get_quantization_attrs().group_sizes,
              (std::vector<uint64_t>{1, 1, std::numeric_limits<uint64_t>::max(), 1}));
    EXPECT_EQ(sdpa->input_value(1).get_element_type(), ov::element::i8);
    EXPECT_EQ(sdpa->input_value(2).get_element_type(), ov::element::i8);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Concat>(sdpa->input_value(1).get_node_shared_ptr()));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Concat>(sdpa->input_value(2).get_node_shared_ptr()));
    const size_t num_data_inputs = sdpa->get_input_size() - sdpa->get_compression_inputs_num();
    EXPECT_EQ(sdpa->input_value(num_data_inputs).get_element_type(), ov::element::f16);
    EXPECT_EQ(sdpa->input_value(num_data_inputs + 1).get_element_type(), ov::element::f16);
    EXPECT_EQ(sdpa->input_value(num_data_inputs).get_partial_shape(), ov::PartialShape({1, kv_num_heads, 1, head_size}));
    EXPECT_EQ(sdpa->input_value(num_data_inputs + 1).get_partial_shape(), ov::PartialShape({1, kv_num_heads, 1, head_size}));
}

TEST(GQADecompositionTest, compressed_kv_preserves_optional_inputs) {
    GQAConfig cfg;
    cfg.kv_cache_bit_width = 8;
    cfg.kv_quant = QuantType::PER_CHANNEL;
    cfg.out_quant = QuantType::PER_CHANNEL;
    cfg.key_scale_shape = ov::PartialShape{kv_num_heads * head_size};
    cfg.value_scale_shape = ov::PartialShape{kv_num_heads * head_size};
    cfg.scale = 0.125f;
    cfg.attention_bias = true;
    cfg.head_sink = true;

    const auto sdpa = decompose_and_get_sdpa(cfg);

    ASSERT_NE(sdpa, nullptr);
    ASSERT_TRUE(sdpa->get_kv_compressed());
    ASSERT_EQ(sdpa->get_input_size(), 8u) << "Q, K, V, mask, scale, sink, K scale, V scale";
    EXPECT_TRUE(slot_holds_a_mask(sdpa->input_value(3)));
    EXPECT_EQ(sdpa->input_value(4).get_element_type(), ov::element::f32);
    EXPECT_EQ(sdpa->input_value(5).get_element_type(), ov::element::f32);
    EXPECT_EQ(sdpa->input_value(6).get_element_type(), ov::element::f16);
    EXPECT_EQ(sdpa->input_value(7).get_element_type(), ov::element::f16);
}

TEST(GQADecompositionTest, int4_i8_cache_uses_u4_zp8) {
    GQAConfig cfg;
    cfg.kv_cache_bit_width = 4;
    cfg.kv_quant = QuantType::PER_CHANNEL;
    cfg.out_quant = QuantType::PER_CHANNEL;
    cfg.key_scale_shape = ov::PartialShape{kv_num_heads * head_size};
    cfg.value_scale_shape = ov::PartialShape{kv_num_heads * head_size};

    const auto sdpa = decompose_and_get_sdpa(cfg);

    ASSERT_NE(sdpa, nullptr);
    ASSERT_TRUE(sdpa->get_kv_compressed());
    ASSERT_EQ(sdpa->get_input_size(), 7u) << "Q, K, V, K scale, V scale, K zero point, V zero point";
    EXPECT_EQ(sdpa->get_quantization_attrs().quantization_type,
              ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric);
    EXPECT_EQ(sdpa->get_quantization_attrs().quantization_dt, ov::element::u4);
    EXPECT_EQ(sdpa->input_value(1).get_element_type(), ov::element::i8);
    EXPECT_EQ(sdpa->input_value(2).get_element_type(), ov::element::i8);
    expect_fixed_int4_zero_point(sdpa->input_value(5));
    expect_fixed_int4_zero_point(sdpa->input_value(6));
}

TEST(GQADecompositionTest, int4_u8_cache_uses_u4_zp8) {
    GQAConfig cfg;
    cfg.kv_cache_bit_width = 4;
    cfg.kv_quant = QuantType::PER_CHANNEL;
    cfg.out_quant = QuantType::PER_CHANNEL;
    cfg.key_scale_shape = ov::PartialShape{kv_num_heads * head_size};
    cfg.value_scale_shape = ov::PartialShape{kv_num_heads * head_size};
    cfg.cache_type = ov::element::u8;

    const auto sdpa = decompose_and_get_sdpa(cfg);

    ASSERT_NE(sdpa, nullptr);
    ASSERT_TRUE(sdpa->get_kv_compressed());
    ASSERT_EQ(sdpa->get_input_size(), 7u) << "Q, K, V, K scale, V scale, K zero point, V zero point";
    EXPECT_EQ(sdpa->get_quantization_attrs().quantization_type,
              ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric);
    EXPECT_EQ(sdpa->get_quantization_attrs().quantization_dt, ov::element::u4);
    EXPECT_EQ(sdpa->input_value(1).get_element_type(), ov::element::u8);
    EXPECT_EQ(sdpa->input_value(2).get_element_type(), ov::element::u8);
    expect_fixed_int4_zero_point(sdpa->input_value(5));
    expect_fixed_int4_zero_point(sdpa->input_value(6));
}

// A sliding-window cache retains the explicit attention mask.
// A full-length static KV cache has unused tail slots: LOWER_RIGHT alignment would expose them, so the mask is kept.
TEST(GQADecompositionTest, static_kv_cache_keeps_mask) {
    GQAConfig cfg;
    cfg.past_len = 8;

    const auto sdpa = decompose_and_get_sdpa(cfg);
    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_size(), 4u) << "Q, K, V, mask";
    EXPECT_TRUE(slot_holds_a_mask(sdpa->input_value(3)));
    EXPECT_FALSE(sdpa->get_causal());
}

TEST(GQADecompositionTest, sliding_window_keeps_mask) {
    GQAConfig cfg;
    cfg.local_window_size = 128;
    cfg.sliding_window_cache = true;

    const auto sdpa = decompose_and_get_sdpa(cfg);
    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_size(), 4u) << "Q, K, V, mask";
    EXPECT_TRUE(slot_holds_a_mask(sdpa->input_value(3)));
}

// Causal smooth softmax retains the explicit mask.
TEST(GQADecompositionTest, smooth_softmax_keeps_mask) {
    GQAConfig cfg;
    cfg.smooth_softmax = true;
    cfg.causal = true;
    cfg.scale = 0.0f;

    const auto sdpa = decompose_and_get_sdpa(cfg);
    ASSERT_NE(sdpa, nullptr);

    // v13::SDPA slots are positional: 3=attn_mask, 4=scale, 5=sink.
    ASSERT_EQ(sdpa->get_input_size(), 6u) << "expected Q,K,V,mask,scale,sink. Got " << sdpa->get_input_size()
                                          << " inputs -- the attn_mask slot was skipped while scale and sink were still "
                                             "appended, so each optional input moved down one position and the sink fell "
                                             "off the end.";

    EXPECT_TRUE(slot_holds_a_mask(sdpa->input_value(3)));

    const bool has_real_mask = sdpa->get_input_size() > 3 && slot_holds_a_mask(sdpa->input_value(3));
    EXPECT_TRUE(sdpa->get_causal() || has_real_mask) << "SDPA is neither is_causal nor masked: full bidirectional attention, "
                                                        "every query position can see future tokens.";
}

// An explicit scale and smooth_softmax retains the explicit mask.
TEST(GQADecompositionTest, smooth_softmax_with_scale_keeps_mask) {
    GQAConfig cfg;
    cfg.smooth_softmax = true;
    cfg.causal = true;
    cfg.scale = 0.125f;

    const auto sdpa = decompose_and_get_sdpa(cfg);
    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_size(), 6u);
    EXPECT_TRUE(slot_holds_a_mask(sdpa->input_value(3)));
}

// A local window with a plain KV cache retains the explicit mask.
TEST(GQADecompositionTest, local_window_keeps_mask) {
    GQAConfig cfg;
    cfg.causal = true;
    cfg.local_window_size = 128;
    cfg.sliding_window_cache = false;  // plain past/present cache, not rotating

    const auto sdpa = decompose_and_get_sdpa(cfg);
    ASSERT_NE(sdpa, nullptr);

    EXPECT_GT(sdpa->get_input_size(), 3u) << "local_window_size=" << cfg.local_window_size
                                          << " requires an explicit window mask, but the mask was elided. is_causal alone "
                                             "cannot express a left window bound, so attention now spans the whole KV cache.";
}

// Combining a local window and sink retains the explicit mask.
TEST(GQADecompositionTest, local_window_with_sink_keeps_mask) {
    GQAConfig cfg;
    cfg.causal = true;
    cfg.smooth_softmax = true;
    cfg.local_window_size = 128;
    cfg.sliding_window_cache = false;

    const auto sdpa = decompose_and_get_sdpa(cfg);
    ASSERT_NE(sdpa, nullptr);

    const bool has_real_mask = sdpa->get_input_size() > 3 && slot_holds_a_mask(sdpa->input_value(3));
    EXPECT_TRUE(has_real_mask) << "windowed + sink layer ended up with no attention mask at all";
    EXPECT_EQ(sdpa->get_input_size(), 6u);
}

}  // namespace
}  // namespace ov::test::intel_gpu

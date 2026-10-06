// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/group_query_attention_decomposition.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <set>
#include <vector>

#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/op/stateless_kv.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
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
    bool static_past_cache = false;
    bool transpose_v = false;
    bool sliding_window_cache = false;
    bool smooth_softmax = false;  // adds an extra logit -> sink branch
    bool attention_bias = false;
    bool head_sink = false;
    bool causal = true;
    ov::PartialShape key_scale_shape{1};
    ov::PartialShape value_scale_shape{1};
};

std::shared_ptr<ov::Model> make_gqa_model(const GQAConfig& cfg) {
    const auto f32 = ov::element::f32;
    const auto past_len = cfg.static_past_cache ? ov::Dimension{128} : ov::Dimension::dynamic();

    auto query = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, num_heads, 1, head_size});
    auto key = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, 1, head_size});
    auto value = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, 1, head_size});
    const auto cache_type = cfg.kv_cache_bit_width ? cfg.cache_type : f32;
    const auto cache_head_size = cfg.kv_cache_bit_width == 4 ? head_size / 2 : head_size;
    auto past_key = std::make_shared<ov::op::v0::Parameter>(cache_type, ov::PartialShape{1, kv_num_heads, past_len, cache_head_size});
    const auto past_value_shape =
        cfg.transpose_v ? ov::PartialShape{1, kv_num_heads, cache_head_size, past_len} : ov::PartialShape{1, kv_num_heads, past_len, cache_head_size};
    auto past_value = std::make_shared<ov::op::v0::Parameter>(cache_type, past_value_shape);
    ov::Output<ov::Node> gqa_past_value = past_value->output(0);
    if (cfg.transpose_v) {
        const auto value_transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
        gqa_past_value = std::make_shared<ov::op::v1::Transpose>(gqa_past_value, value_transpose_order);
    }
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});

    ov::OutputVector inputs(14);
    inputs[0] = query;
    inputs[1] = key;
    inputs[2] = value;
    inputs[3] = past_key;
    inputs[4] = gqa_past_value;
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
    for (size_t output_idx = 0; output_idx < gqa->get_output_size(); ++output_idx) {
        auto output = gqa->output(output_idx);
        if (cfg.transpose_v && output_idx == 2) {
            const auto value_transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
            output = std::make_shared<ov::op::v1::Transpose>(output, value_transpose_order);
        }
        results.push_back(std::make_shared<ov::op::v0::Result>(output));
    }
    return std::make_shared<ov::Model>(results, parameters);
}

std::shared_ptr<ov::Model> decompose_gqa_model(const GQAConfig& cfg) {
    auto model = make_gqa_model(cfg);
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::GroupQueryAttentionDecomposition>();
    manager.run_passes(model);
    return model;
}

void expect_sdpa_kv_not_broadcast(const std::shared_ptr<ov::intel_gpu::op::SDPA>& sdpa) {
    for (size_t input_index = 1; input_index <= 2; ++input_index) {
        const auto& kv_shape = sdpa->input_value(input_index).get_partial_shape();
        ASSERT_TRUE(kv_shape.rank().is_static());
        ASSERT_GE(kv_shape.rank().get_length(), 2);
        EXPECT_EQ(kv_shape[1], ov::Dimension(kv_num_heads)) << "SDPA input " << input_index << " has broadcasted KV heads";
    }
}

std::shared_ptr<ov::intel_gpu::op::SDPA> decompose_and_get_sdpa(const GQAConfig& cfg) {
    auto model = decompose_gqa_model(cfg);

    std::shared_ptr<ov::intel_gpu::op::SDPA> result;
    for (const auto& node : model->get_ordered_ops()) {
        EXPECT_FALSE(ov::is_type<ov::op::internal::GroupQueryAttention>(node));
        if (auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SDPA>(node)) {
            result = sdpa;
        }
    }
    EXPECT_NE(result, nullptr);
    if (result) {
        expect_sdpa_kv_not_broadcast(result);
    }
    return result;
}

void expect_stateless_kv_cache_connections(const GQAConfig& cfg) {
    const auto model = decompose_gqa_model(cfg);
    const auto parameters = model->get_parameters();
    EXPECT_EQ(parameters[3]->get_partial_shape().is_static(), cfg.static_past_cache);

    std::shared_ptr<ov::intel_gpu::op::SDPA> sdpa;
    std::vector<std::shared_ptr<ov::intel_gpu::op::StatelessKV>> statelesskvs;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto current_sdpa = ov::as_type_ptr<ov::intel_gpu::op::SDPA>(node)) {
            ASSERT_EQ(sdpa, nullptr);
            sdpa = current_sdpa;
        }
        if (auto stateless_kv = ov::as_type_ptr<ov::intel_gpu::op::StatelessKV>(node)) {
            statelesskvs.emplace_back(std::move(stateless_kv));
        }
    }

    ASSERT_NE(sdpa, nullptr);
    expect_sdpa_kv_not_broadcast(sdpa);
    ASSERT_EQ(statelesskvs.size(), 2u);

    std::shared_ptr<ov::intel_gpu::op::StatelessKV> key;
    std::shared_ptr<ov::intel_gpu::op::StatelessKV> value;

    const auto find_sdpa_index = [&](const std::set<ov::Input<ov::Node>>& targets, const auto& self) -> std::optional<size_t> {
        for (const auto& target : targets) {
            if (ov::is_type<ov::op::v0::ShapeOf>(target.get_node())) {
                continue;
            } else if (target.get_node() == sdpa.get()) {
                return target.get_index();
            }
        }
        for (const auto& target : targets) {
            for (size_t i = 0; i < target.get_node()->get_output_size(); ++i) {
                const auto targets_ = target.get_node()->get_output_target_inputs(i);
                const auto result = self(targets_, self);
                if (result) {
                    return result;
                }
            }
        }
        return std::nullopt;
    };

    for (const auto& stateless_kv : statelesskvs) {
        const auto sdpa_index = find_sdpa_index(stateless_kv->output(1).get_target_inputs(), find_sdpa_index);
        if (sdpa_index == 1) {
            ASSERT_EQ(key, nullptr);
            key = stateless_kv;
        } else if (sdpa_index == 2) {
            ASSERT_EQ(value, nullptr);
            value = stateless_kv;
        } else {
            ASSERT_TRUE(sdpa_index == 1 || sdpa_index == 2);
        }
    }

    const auto present_key_source = model->output(1).get_node()->input_value(0);
    const auto present_value_source = model->output(2).get_node()->input_value(0);
    ASSERT_EQ(present_key_source, key->output(0));
    ASSERT_EQ(present_value_source, value->output(0));
    EXPECT_EQ(key->get_concat_axis(), 2);
    EXPECT_EQ(value->get_concat_axis(), cfg.transpose_v ? 3 : 2);
    EXPECT_EQ(key->input_value(0).get_node_shared_ptr(), parameters[3]);
    EXPECT_EQ(key->input_value(1).get_node_shared_ptr(), parameters[1]);
    if (cfg.transpose_v) {
        const auto value_transpose = ov::as_type_ptr<ov::op::v1::Transpose>(value->input_value(1).get_node_shared_ptr());
        ASSERT_NE(value_transpose, nullptr);
        const auto value_transpose_order = ov::as_type_ptr<ov::op::v0::Constant>(value_transpose->input_value(1).get_node_shared_ptr());
        ASSERT_NE(value_transpose_order, nullptr);
        EXPECT_THAT(value_transpose_order->cast_vector<int64_t>(), ::testing::ElementsAre(0, 1, 3, 2));
        EXPECT_EQ(value_transpose->input_value(0).get_node_shared_ptr(), parameters[2]);
    } else {
        EXPECT_EQ(value->input_value(0).get_node_shared_ptr(), parameters[4]);
        EXPECT_EQ(value->input_value(1).get_node_shared_ptr(), parameters[2]);
    }
    EXPECT_EQ(key->input_value(2).get_node_shared_ptr(), value->input_value(2).get_node_shared_ptr());

    if (cfg.transpose_v) {
        EXPECT_THAT(sdpa->get_input2_transpose_order(), ::testing::ElementsAre(0, 1, 3, 2));
    } else {
        EXPECT_EQ(sdpa->get_input2_transpose_order(), ov::intel_gpu::op::SDPA::default_order(4));
    }
}

TEST(GQADecompositionTest, static_input_uses_stateless_kv) {
    GQAConfig cfg;
    cfg.static_past_cache = true;
    expect_stateless_kv_cache_connections(cfg);
}

TEST(GQADecompositionTest, dynamic_input_uses_stateless_kv) {
    GQAConfig cfg;

    expect_stateless_kv_cache_connections(cfg);
}

TEST(GQADecompositionTest, transposed_value_cache_fuses_transposes) {
    GQAConfig cfg;
    cfg.transpose_v = true;

    expect_stateless_kv_cache_connections(cfg);
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
    EXPECT_TRUE(ov::is_type<ov::intel_gpu::op::StatelessKV>(sdpa->input_value(1).get_node_shared_ptr()));
    EXPECT_TRUE(ov::is_type<ov::intel_gpu::op::StatelessKV>(sdpa->input_value(2).get_node_shared_ptr()));
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

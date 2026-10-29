// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gqa_compiled_model.hpp"
#include "intel_npu/config/npuw.hpp"
#include "llm_test_helpers.hpp"
#include "npuw_transformations/reshape_to_static_gqa.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "serialization.hpp"

namespace {

using ov::test::npuw::MockSubCompiledModel;
using ov::test::npuw::NullPlugin;
using ov::test::npuw::build_llm_test_model;

template <class Op>
std::size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ops();
    return std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<Op>(op);
    });
}

std::shared_ptr<ov::Model> build_group_query_attention_model() {
    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 1, 16});
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 8, 16});
    auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 8, 16});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});

    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
        ov::OutputVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length},
        4,
        2,
        0.0f,
        false,
        false);

    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(gqa->output(0)),
                                std::make_shared<ov::op::v0::Result>(gqa->output(1)),
                                std::make_shared<ov::op::v0::Result>(gqa->output(2))};
    ov::ParameterVector params = {query, key, value, past_key, past_value, seqlens_k, total_sequence_length};
    return std::make_shared<ov::Model>(results, params, "gqa_model");
}

// Builds a model with a well-formed GQA op plus the surrounding transformer traits
// (activation input, position_ids, named past/present KV cache Parameters/Results)
// that supports() looks for. `position_signal` picks between an explicit `position_ids`
// Parameter (V1) or a `past_seq_len`/`total_seq_len` pair with no position_ids input (V0).
enum class PositionSignal {
    PositionIds,
    SeqLenPair,
};

std::shared_ptr<ov::Model> build_full_gqa_transformer_model(int64_t num_heads = 4,
                                                            int64_t kv_num_heads = 2,
                                                            bool do_rotary = true,
                                                            PositionSignal position_signal = PositionSignal::PositionIds,
                                                            bool with_2bit_weights = false) {
    // Shaped {1, num_heads, 16} so it Unsqueeze's below into QUERY's {1, num_heads, 1, 16}
    // without a volume mismatch -- identify_case() requires this Parameter to be
    // data-connected to the GQA op, not just present by name.
    auto input_hidden_states =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, static_cast<size_t>(num_heads), 16});
    input_hidden_states->set_friendly_name("input_hidden_states");
    auto query_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    auto query = std::make_shared<ov::op::v0::Unsqueeze>(input_hidden_states, query_axis);

    std::shared_ptr<ov::op::v0::Parameter> position_ids;
    if (position_signal == PositionSignal::PositionIds) {
        position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 4});
        position_ids->set_friendly_name("position_ids");
    }

    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::Shape{1, static_cast<size_t>(kv_num_heads), 1, 16});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                         ov::Shape{1, static_cast<size_t>(kv_num_heads), 1, 16});
    auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                            ov::Shape{1, static_cast<size_t>(kv_num_heads), 8, 16});
    past_key->set_friendly_name("past_keys_0");
    auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                              ov::Shape{1, static_cast<size_t>(kv_num_heads), 8, 16});
    past_value->set_friendly_name("past_values_0");
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    if (position_signal == PositionSignal::SeqLenPair) {
        // V0 wiring: no position_ids input; seqlens_k/total_sequence_length double as the
        // "past_seq_len"/"total_seq_len" naming evidence.
        seqlens_k->set_friendly_name("past_seq_len");
        total_sequence_length->set_friendly_name("total_seq_len");
    }
    auto cos_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});
    auto sin_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});

    ov::OutputVector gqa_inputs{query, key, value, past_key, past_value, seqlens_k, total_sequence_length,
                               cos_cache, sin_cache};
    if (position_ids) {
        // POSITION_IDS (index 9) -- wiring it here is what identify_case() requires as evidence.
        gqa_inputs.push_back(position_ids);
    }
    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(gqa_inputs,
                                                                       num_heads,
                                                                       kv_num_heads,
                                                                       0.0f,
                                                                       do_rotary,
                                                                       false);

    auto present_key = std::make_shared<ov::op::v0::Result>(gqa->output(1));
    present_key->set_friendly_name("present_keys_0");
    auto present_value = std::make_shared<ov::op::v0::Result>(gqa->output(2));
    present_value->set_friendly_name("present_values_0");

    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(gqa->output(0)), present_key, present_value};
    ov::ParameterVector params = {input_hidden_states};
    if (position_ids) {
        params.push_back(position_ids);
    }
    params.insert(params.end(),
                 {key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache, sin_cache});
    if (with_2bit_weights) {
        // V2 models ship weights as 2-bit group-quantized (u2) Parameters. The GQA op
        // doesn't need to consume this directly; has_2bit_group_quantized_weights() just
        // scans the model's Parameters.
        auto quantized_weight = std::make_shared<ov::op::v0::Parameter>(ov::element::u2, ov::Shape{8, 16});
        quantized_weight->set_friendly_name("quantized_weight");
        params.push_back(quantized_weight);
    }
    return std::make_shared<ov::Model>(results, params, "gqa_full_transformer_model");
}

// Leaves past_key/past_value's KV-cache dimension dynamic at `axis` (2 plain [N,H,S,E],
// 3 transpose_v [N,H,E,S]) to exercise has_dynamic_max_seq_len()/prepare().
void make_kv_cache_dynamic(const std::shared_ptr<ov::Model>& model, size_t axis) {
    for (const auto& parameter : model->get_parameters()) {
        const auto& name = parameter->get_friendly_name();
        if (name != "past_keys_0" && name != "past_values_0") {
            continue;
        }
        auto shape = parameter->get_partial_shape();
        shape[axis] = ov::Dimension::dynamic();
        parameter->set_partial_shape(shape);
    }
    model->validate_nodes_and_infer_types();
}

// Same wiring as build_full_gqa_transformer_model(), but with the KV-cache dimension left
// dynamic at `axis` to exercise has_dynamic_max_seq_len()/prepare().
std::shared_ptr<ov::Model> build_gqa_model_with_dynamic_kv_cache(size_t axis) {
    auto model = build_full_gqa_transformer_model();
    make_kv_cache_dynamic(model, axis);
    return model;
}

// Same wiring, but with an extra `attention_mask` feeding ATTENTION_BIAS (index 10) with
// a dynamic last dimension, mirroring speculative-decode dumps. POSITION_IDS (index 9)
// is "not provided" via a zero-size Constant, matching the V0 wiring.
std::shared_ptr<ov::Model> build_gqa_model_with_dynamic_attention_bias(bool dynamic_kv_cache = false) {
    auto input_hidden_states = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 16});
    input_hidden_states->set_friendly_name("input_hidden_states");

    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 1, 16});
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});

    ov::PartialShape kv_shape{1, 2, 8, 16};
    if (dynamic_kv_cache) {
        kv_shape[2] = ov::Dimension::dynamic();
    }
    auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, kv_shape);
    past_key->set_friendly_name("past_keys_0");
    auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, kv_shape);
    past_value->set_friendly_name("past_values_0");

    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    seqlens_k->set_friendly_name("past_seq_len");
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    total_sequence_length->set_friendly_name("total_seq_len");
    auto cos_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});
    auto sin_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});

    auto position_ids_placeholder = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{0}, {});
    auto attention_mask =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 1, 4, ov::Dimension::dynamic()});
    attention_mask->set_friendly_name("attention_mask");

    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
        ov::OutputVector{query,
                        key,
                        value,
                        past_key,
                        past_value,
                        seqlens_k,
                        total_sequence_length,
                        cos_cache,
                        sin_cache,
                        position_ids_placeholder,
                        attention_mask},
        4,
        2,
        0.0f,
        true,
        false);

    auto present_key = std::make_shared<ov::op::v0::Result>(gqa->output(1));
    present_key->set_friendly_name("present_keys_0");
    auto present_value = std::make_shared<ov::op::v0::Result>(gqa->output(2));
    present_value->set_friendly_name("present_values_0");

    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(gqa->output(0)), present_key, present_value};
    ov::ParameterVector params = {input_hidden_states,
                                 query,
                                 key,
                                 value,
                                 past_key,
                                 past_value,
                                 seqlens_k,
                                 total_sequence_length,
                                 cos_cache,
                                 sin_cache,
                                 attention_mask};
    auto model = std::make_shared<ov::Model>(results, params, "gqa_attention_bias_model");
    model->validate_nodes_and_infer_types();
    return model;
}

// Same wiring as build_gqa_model_with_dynamic_kv_cache(), but with sliding_window_cache=
// true -- a dynamic axis here must be pinned by ReshapeToStaticGQA to its own
// local_window_size-derived capacity, not the model-wide max_seq_len.
std::shared_ptr<ov::Model> build_gqa_model_with_sliding_window_cache(size_t axis) {
    auto input_hidden_states = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 16});
    input_hidden_states->set_friendly_name("input_hidden_states");
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 4});
    position_ids->set_friendly_name("position_ids");

    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 1, 16});
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});

    ov::PartialShape kv_shape{1, 2, 8, 16};
    kv_shape[axis] = ov::Dimension::dynamic();
    auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, kv_shape);
    past_key->set_friendly_name("past_keys_0");
    auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, kv_shape);
    past_value->set_friendly_name("past_values_0");

    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto cos_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});
    auto sin_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});

    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
        ov::OutputVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache, sin_cache},
        /*num_heads=*/4,
        /*kv_num_heads=*/2,
        /*scale=*/0.0f,
        /*do_rotary=*/true,
        /*rotary_interleaved=*/false,
        /*kv_cache_bit_width=*/0,
        ov::op::internal::GroupQueryAttentionQuantType::NONE,
        ov::op::internal::GroupQueryAttentionQuantType::NONE,
        /*local_window_size=*/6,
        /*sliding_window_cache=*/true);

    auto present_key = std::make_shared<ov::op::v0::Result>(gqa->output(1));
    present_key->set_friendly_name("present_keys_0");
    auto present_value = std::make_shared<ov::op::v0::Result>(gqa->output(2));
    present_value->set_friendly_name("present_values_0");

    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(gqa->output(0)), present_key, present_value};
    ov::ParameterVector params = {input_hidden_states,
                                 position_ids,
                                 query,
                                 key,
                                 value,
                                 past_key,
                                 past_value,
                                 seqlens_k,
                                 total_sequence_length,
                                 cos_cache,
                                 sin_cache};
    auto model = std::make_shared<ov::Model>(results, params, "gqa_sliding_window_cache_model");
    model->validate_nodes_and_infer_types();
    return model;
}

// A single model with TWO independent GQA layers sharing the same dynamic axis: a
// "global" layer (sliding_window_cache=false, should take the model-wide max_seq_len)
// and a "sliding" layer (sliding_window_cache=true, local_window_size=6, should get its
// own small capacity). Proves the two don't bleed into each other.
std::shared_ptr<ov::Model> build_gqa_model_with_mixed_global_and_sliding_layers() {
    auto make_layer = [](const std::string& suffix, bool sliding) {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 1, 16});
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});

        auto past_key =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 2, ov::Dimension::dynamic(), 16});
        past_key->set_friendly_name("past_keys_" + suffix);
        auto past_value =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 2, ov::Dimension::dynamic(), 16});
        past_value->set_friendly_name("past_values_" + suffix);

        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
        auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
        auto cos_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});
        auto sin_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});

        auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
            ov::OutputVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache, sin_cache},
            /*num_heads=*/4,
            /*kv_num_heads=*/2,
            /*scale=*/0.0f,
            /*do_rotary=*/true,
            /*rotary_interleaved=*/false,
            /*kv_cache_bit_width=*/0,
            ov::op::internal::GroupQueryAttentionQuantType::NONE,
            ov::op::internal::GroupQueryAttentionQuantType::NONE,
            /*local_window_size=*/sliding ? 6 : -1,
            /*sliding_window_cache=*/sliding);

        auto present_key = std::make_shared<ov::op::v0::Result>(gqa->output(1));
        present_key->set_friendly_name("present_keys_" + suffix);
        auto present_value = std::make_shared<ov::op::v0::Result>(gqa->output(2));
        present_value->set_friendly_name("present_values_" + suffix);

        ov::ResultVector layer_results = {std::make_shared<ov::op::v0::Result>(gqa->output(0)),
                                          present_key,
                                          present_value};
        ov::ParameterVector layer_params =
            {query, key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache, sin_cache};
        return std::make_pair(layer_results, layer_params);
    };

    auto [global_results, global_params] = make_layer("global", /*sliding=*/false);
    auto [sliding_results, sliding_params] = make_layer("sliding", /*sliding=*/true);

    ov::ResultVector results = global_results;
    results.insert(results.end(), sliding_results.begin(), sliding_results.end());
    ov::ParameterVector params = global_params;
    params.insert(params.end(), sliding_params.begin(), sliding_params.end());

    auto model = std::make_shared<ov::Model>(results, params, "gqa_mixed_global_and_sliding_model");
    model->validate_nodes_and_infer_types();
    return model;
}

// Same two-layer wiring as build_gqa_model_with_mixed_global_and_sliding_layers(), but
// neither GQA op sets sliding_window_cache/local_window_size; instead the model carries
// rt_info["framework"]["sliding_window_layers"/"sliding_window_size"] (mirroring ONNX
// metadata_props). The "sliding" layer's query is wired via a real data dependency on the
// "global" layer's output, so topological op order matches layer numbering (0, 1, ...) --
// the same assumption index_gqa_layers() relies on. Unconnected branches would leave
// relative order implementation-defined and make this test flaky.
std::shared_ptr<ov::Model> build_gqa_model_with_rt_info_sliding_window_layers(size_t declared_sliding_window_size) {
    auto make_layer = [](const std::string& suffix, const ov::Output<ov::Node>& prev_attention_output) {
        ov::Output<ov::Node> query_input;
        auto query_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 1, 16});
        if (prev_attention_output.get_node()) {
            // Previous layer's output is [1, 1, 64]; reshape to [1, 4, 1, 16] and fold
            // into this layer's query so this GQA op depends on the previous one.
            auto reshaped_prev = std::make_shared<ov::op::v1::Reshape>(
                prev_attention_output,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 4, 1, 16}),
                false);
            query_input = std::make_shared<ov::op::v1::Add>(query_param, reshaped_prev);
        } else {
            query_input = query_param;
        }

        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});

        auto past_key =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 2, ov::Dimension::dynamic(), 16});
        past_key->set_friendly_name("past_keys_" + suffix);
        auto past_value =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 2, ov::Dimension::dynamic(), 16});
        past_value->set_friendly_name("past_values_" + suffix);

        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
        auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
        auto cos_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});
        auto sin_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});

        auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
            ov::OutputVector{query_input, key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache, sin_cache},
            /*num_heads=*/4,
            /*kv_num_heads=*/2,
            /*scale=*/0.0f,
            /*do_rotary=*/true,
            /*rotary_interleaved=*/false,
            /*kv_cache_bit_width=*/0,
            ov::op::internal::GroupQueryAttentionQuantType::NONE,
            ov::op::internal::GroupQueryAttentionQuantType::NONE);

        auto present_key = std::make_shared<ov::op::v0::Result>(gqa->output(1));
        present_key->set_friendly_name("present_keys_" + suffix);
        auto present_value = std::make_shared<ov::op::v0::Result>(gqa->output(2));
        present_value->set_friendly_name("present_values_" + suffix);

        ov::ResultVector layer_results = {std::make_shared<ov::op::v0::Result>(gqa->output(0)),
                                          present_key,
                                          present_value};
        ov::ParameterVector layer_params = {query_param,
                                            key,
                                            value,
                                            past_key,
                                            past_value,
                                            seqlens_k,
                                            total_sequence_length,
                                            cos_cache,
                                            sin_cache};
        return std::make_tuple(layer_results, layer_params, gqa->output(0));
    };

    auto [global_results, global_params, global_attention_output] = make_layer("global", ov::Output<ov::Node>());
    auto [sliding_results, sliding_params, sliding_attention_output] = make_layer("sliding", global_attention_output);

    ov::ResultVector results = global_results;
    results.insert(results.end(), sliding_results.begin(), sliding_results.end());
    ov::ParameterVector params = global_params;
    params.insert(params.end(), sliding_params.begin(), sliding_params.end());

    auto model = std::make_shared<ov::Model>(results, params, "gqa_rt_info_sliding_window_model");
    model->set_rt_info(std::string("[1]"), "framework", "sliding_window_layers");
    model->set_rt_info(std::string("[0]"), "framework", "full_attention_layers");
    model->set_rt_info(std::to_string(declared_sliding_window_size), "framework", "sliding_window_size");
    model->validate_nodes_and_infer_types();
    return model;
}

// Same two-independent-layer shape, but with neither sliding_window_cache/
// local_window_size attributes nor rt_info -- only each layer's SEQLENS_K/
// TOTAL_SEQUENCE_LENGTH Parameter name optionally hints "sliding". When `name_hint` is
// false, no hint exists and the fallback must leave every layer at max_seq_len.
std::shared_ptr<ov::Model> build_gqa_model_with_seqlens_name_hint(bool name_hint) {
    auto make_layer = [&](const std::string& suffix, bool sliding) {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 1, 16});
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});

        auto past_key =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 2, ov::Dimension::dynamic(), 16});
        past_key->set_friendly_name("past_keys_" + suffix);
        auto past_value =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 2, ov::Dimension::dynamic(), 16});
        past_value->set_friendly_name("past_values_" + suffix);

        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
        auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
        if (name_hint) {
            seqlens_k->set_friendly_name(sliding ? "past_seq_len_sliding" : "past_seq_len_full");
            total_sequence_length->set_friendly_name(sliding ? "total_seq_len_sliding" : "total_seq_len_full");
        }
        auto cos_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});
        auto sin_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});

        auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
            ov::OutputVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache, sin_cache},
            /*num_heads=*/4,
            /*kv_num_heads=*/2,
            /*scale=*/0.0f,
            /*do_rotary=*/true,
            /*rotary_interleaved=*/false,
            /*kv_cache_bit_width=*/0,
            ov::op::internal::GroupQueryAttentionQuantType::NONE,
            ov::op::internal::GroupQueryAttentionQuantType::NONE);

        auto present_key = std::make_shared<ov::op::v0::Result>(gqa->output(1));
        present_key->set_friendly_name("present_keys_" + suffix);
        auto present_value = std::make_shared<ov::op::v0::Result>(gqa->output(2));
        present_value->set_friendly_name("present_values_" + suffix);

        ov::ResultVector layer_results = {std::make_shared<ov::op::v0::Result>(gqa->output(0)),
                                          present_key,
                                          present_value};
        ov::ParameterVector layer_params =
            {query, key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache, sin_cache};
        return std::make_pair(layer_results, layer_params);
    };

    auto [global_results, global_params] = make_layer("global", /*sliding=*/false);
    auto [sliding_results, sliding_params] = make_layer("sliding", /*sliding=*/true);

    ov::ResultVector results = global_results;
    results.insert(results.end(), sliding_results.begin(), sliding_results.end());
    ov::ParameterVector params = global_params;
    params.insert(params.end(), sliding_params.begin(), sliding_params.end());

    auto model = std::make_shared<ov::Model>(results, params, "gqa_seqlens_name_hint_model");
    model->validate_nodes_and_infer_types();
    return model;
}

std::shared_ptr<ov::Model> build_unqdq_model(const ov::element::Type& input_type = ov::element::f32) {
    auto input = std::make_shared<ov::op::v0::Parameter>(input_type, ov::Shape{1, 4});
    auto input_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto input_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto output_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto output_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {255.0f});
    auto fake_quantize =
        std::make_shared<ov::op::v0::FakeQuantize>(input, input_low, input_high, output_low, output_high, 256);
    auto quantized_convert = std::make_shared<ov::op::v0::Convert>(fake_quantize, ov::element::u16);
    auto dequantized_convert = std::make_shared<ov::op::v0::Convert>(quantized_convert, ov::element::f32);
    auto zero_point = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::u16, ov::Shape{}, {128}),
        ov::element::f32);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(dequantized_convert, zero_point);
    auto scale = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.1f}),
        ov::element::f32);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(multiply)},
                                       ov::ParameterVector{input},
                                       "gqa_unqdq_model");
}

std::shared_ptr<ov::Model> build_hidden_states_model(std::size_t tokens) {
    auto input_hidden_states =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, tokens, 16});
    input_hidden_states->set_friendly_name("input_hidden_states");
    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(input_hidden_states)},
        ov::ParameterVector{input_hidden_states},
        "gqa_hidden_states_model");
}

std::shared_ptr<ov::Model> build_conv_to_matmul_model() {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 8, 3});
    auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{4, 3, 1, 1});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{4, 1, 1, 1});

    auto transpose_in = std::make_shared<ov::op::v1::Transpose>(
        activation,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 3, 1, 2}));
    auto scaled_weights = std::make_shared<ov::op::v1::Multiply>(
        std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32),
        std::make_shared<ov::op::v0::Convert>(scale, ov::element::f32));
    auto convolution = std::make_shared<ov::op::v1::Convolution>(transpose_in,
                                                                 scaled_weights,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});
    auto transpose_out = std::make_shared<ov::op::v1::Transpose>(
        convolution,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 2, 3, 1}));

    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(transpose_out)},
                                       ov::ParameterVector{activation, weights, scale},
                                       "gqa_conv_to_matmul_model");
}

std::shared_ptr<ov::Model> build_dumped_gqa_conv_model() {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 512, 5120});
    auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::i4, ov::Shape{5120, 5120, 1, 1});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5120});

    auto converted_activation = std::make_shared<ov::op::v0::Convert>(activation, ov::element::f32);
    auto unsqueezed_activation = std::make_shared<ov::op::v0::Unsqueeze>(
        converted_activation,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    auto transposed_activation = std::make_shared<ov::op::v1::Transpose>(
        unsqueezed_activation,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2}));
    auto scaled_weights = std::make_shared<ov::op::v1::Multiply>(
        std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32),
        std::make_shared<ov::op::v1::Reshape>(
            scale,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {5120, 1, 1, 1}),
            true));
    auto convolution = std::make_shared<ov::op::v1::Convolution>(transposed_activation,
                                                                scaled_weights,
                                                                ov::Strides{1, 1},
                                                                ov::CoordinateDiff{0, 0},
                                                                ov::CoordinateDiff{0, 0},
                                                                ov::Strides{1, 1});
    auto transpose_out = std::make_shared<ov::op::v1::Transpose>(
        convolution,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1}));

    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(transpose_out)},
                                       ov::ParameterVector{activation, weights, scale},
                                       "dumped_gqa_conv_model");
}

std::shared_ptr<ov::Model> build_conv_to_matmul_and_unqdq_model() {
    const auto conv_model = build_conv_to_matmul_model();
    const auto unqdq_model = build_unqdq_model(ov::element::f16);

    ov::ResultVector results;
    for (const auto& result : conv_model->get_results()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(result->input_value(0)));
    }
    for (const auto& result : unqdq_model->get_results()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(result->input_value(0)));
    }

    ov::ParameterVector parameters = conv_model->get_parameters();
    const auto unqdq_parameters = unqdq_model->get_parameters();
    parameters.insert(parameters.end(), unqdq_parameters.begin(), unqdq_parameters.end());

    return std::make_shared<ov::Model>(results, parameters, "gqa_conv_to_matmul_and_unqdq_model");
}

struct CompileCall {
    ov::AnyMap props;
    std::shared_ptr<ov::Model> model;
};

class RecordingFactory {
public:
    ov::npuw::GQACompiledModel::CompiledModelFactory make_factory() {
        return [this](const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel> {
            m_calls.push_back({props, model});
            return std::make_shared<MockSubCompiledModel>(model, plugin, props);
        };
    }

    const CompileCall& only_call() const {
        OPENVINO_ASSERT(m_calls.size() == 1u, "Expected a single compile call");
        return m_calls.front();
    }

private:
    std::vector<CompileCall> m_calls;
};

class PropertyForwardingMockCompiledModel final : public ov::npuw::ICompiledModel {
public:
    PropertyForwardingMockCompiledModel(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::npuw::ICompiledModel(model, plugin) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return {};
    }
    void set_property(const ov::AnyMap& properties) override {
        last_set_properties = properties;
    }
    ov::Any get_property(const std::string& name) const override {
        if (name == "NPUW_FOLD") {
            return true;
        }
        return {};
    }

private:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return {};
    }

public:
    ov::AnyMap last_set_properties;
};


class GQACompiledModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    static ov::AnyMap base_props() {
        return {{"NPUW_GQA", "YES"}};
    }

    static void merge_props(ov::AnyMap& dst, const ov::AnyMap& src) {
        for (const auto& [key, value] : src) {
            dst[key] = value;
        }
    }

    std::unique_ptr<ov::npuw::GQACompiledModel> create_compiled_model(const std::shared_ptr<ov::Model>& model,
                                                                      const ov::AnyMap& extra_props,
                                                                      RecordingFactory& recorder) const {
        auto props = base_props();
        merge_props(props, extra_props);
        return std::make_unique<ov::npuw::GQACompiledModel>(model, m_plugin, props, recorder.make_factory());
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
};

TEST_F(GQACompiledModelTest, AddsExpectedNpuwDefaultsBeforeInnerCompilation) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_test_model(), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "NONE");
    EXPECT_EQ(call.props.at("NPUW_DEVICES").as<std::string>(), "NPU");
    EXPECT_EQ(call.props.at("NPUW_ONLINE_ISOLATE").as<std::string>(), "ATTN");
    EXPECT_EQ(call.props.at("NPUW_ONLINE_KEEP_BLOCKS_TAGGED").as<std::string>(), "attn");
    EXPECT_EQ(call.props.at("NPUW_ATTN").as<std::string>(), "STATIC");
    EXPECT_EQ(call.props.at("NPUW_FOLD").as<std::string>(), "YES");
    EXPECT_EQ(call.props.at(ov::cache_mode.name()).as<ov::CacheMode>(), ov::CacheMode::OPTIMIZE_SPEED);
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "YES");
    EXPECT_EQ(call.props.count("NPUW_FUNCALL_ASYNC"), 0u);
    EXPECT_EQ(call.props.count("NPUW_UNFOLD_IREQS"), 0u);
}

TEST_F(GQACompiledModelTest, DisablesOnlinePipelineForCaseV1Models) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        build_full_gqa_transformer_model(4, 2, true, PositionSignal::PositionIds),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "NONE");
    // V1 doesn't use online partitioning at all, so parallelizing its compilation
    // wouldn't do anything.
    EXPECT_EQ(call.props.count(std::string(::intel_npu::NPUW_PARALLEL_COMPILE::key())), 0u);
}

TEST_F(GQACompiledModelTest, DisablesOnlinePipelineForCaseV0Models) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        build_full_gqa_transformer_model(4, 2, true, PositionSignal::SeqLenPair),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "NONE");
    // V0 doesn't use online partitioning at all, so parallelizing its compilation
    // wouldn't do anything.
    EXPECT_EQ(call.props.count(std::string(::intel_npu::NPUW_PARALLEL_COMPILE::key())), 0u);
}

TEST_F(GQACompiledModelTest, UsesRepOnlinePipelineForCaseV2PrefillModels) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        build_full_gqa_transformer_model(4,
                                                        2,
                                                        true,
                                                        PositionSignal::SeqLenPair,
                                                        /*with_2bit_weights=*/true),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    // Unlike V0/V1, a 2-bit group-quantized (V2) model isn't forced to NONE -- its prefill
    // stage still benefits from online partitioning, so it switches to the REP
    // (repeated-block) pipeline, and compiles the resulting blocks in parallel.
    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "REP");
    EXPECT_EQ(call.props.at(std::string(::intel_npu::NPUW_PARALLEL_COMPILE::key())).as<std::string>(), "YES");
}

TEST_F(GQACompiledModelTest, UsesParallelCompileWithDefaultPipelineForCaseV2GenerateModels) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    // num_heads == kv_num_heads == 1 so input_hidden_states' token dim is 1 -> generate stage.
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        build_full_gqa_transformer_model(1,
                                                        1,
                                                        true,
                                                        PositionSignal::SeqLenPair,
                                                        /*with_2bit_weights=*/true),
                        {},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    // REP is a prefill-only optimization -- the generate stage keeps the plugin-wide
    // default pipeline (REG), but still compiles in parallel like the prefill stage.
    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.count("NPUW_ONLINE_PIPELINE"), 0u);
    EXPECT_EQ(call.props.at(std::string(::intel_npu::NPUW_PARALLEL_COMPILE::key())).as<std::string>(), "YES");
}

TEST_F(GQACompiledModelTest, AppliesFoldOnlyAttnForGenerateStyleModels) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hidden_states_model(1), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "NONE");
    EXPECT_EQ(call.props.at("NPUW_DEVICES").as<std::string>(), "NPU");
    EXPECT_EQ(call.props.at("NPUW_FOLD").as<std::string>(), "YES");
    EXPECT_EQ(call.props.at(ov::cache_mode.name()).as<ov::CacheMode>(), ov::CacheMode::OPTIMIZE_SPEED);
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "YES");
    EXPECT_EQ(call.props.at("NPUW_FUNCALL_ASYNC").as<std::string>(), "YES");
    EXPECT_EQ(call.props.at("NPUW_UNFOLD_IREQS").as<std::string>(), "YES");
    EXPECT_EQ(call.props.at("NPUW_FOLD_ONLY").as<std::string>(), "attn");
    EXPECT_EQ(call.props.count("NPUW_ONLINE_ISOLATE"), 0u);
    EXPECT_EQ(call.props.count("NPUW_ONLINE_KEEP_BLOCKS_TAGGED"), 0u);
    EXPECT_EQ(call.props.count("NPUW_ATTN"), 0u);
}

TEST_F(GQACompiledModelTest, KeepsAttnIsolationDefaultsForPrefillStyleModels) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hidden_states_model(8), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_DEVICES").as<std::string>(), "NPU");
    EXPECT_EQ(call.props.at("NPUW_ONLINE_ISOLATE").as<std::string>(), "ATTN");
    EXPECT_EQ(call.props.at("NPUW_FOLD").as<std::string>(), "YES");
    EXPECT_EQ(call.props.count("NPUW_FUNCALL_ASYNC"), 0u);
    EXPECT_EQ(call.props.count("NPUW_UNFOLD_IREQS"), 0u);
}

TEST_F(GQACompiledModelTest, KeepsUserProvidedLowLevelOverrides) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_test_model(),
                                                     {{"NPUW_ONLINE_PIPELINE", "REG"},
                                                      {"NPUW_DEVICES", "CPU"},
                                                      {"NPUW_ONLINE_ISOLATE", "COMPUTE"},
                                                      {"NPUW_ATTN", "STATIC"},
                                                      {"NPUW_FOLD", "NO"},
                                                      {"NPUW_FUNCALL_ASYNC", "NO"},
                                                      {"NPUW_UNFOLD_IREQS", "NO"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "REG");
    EXPECT_EQ(call.props.at("NPUW_DEVICES").as<std::string>(), "CPU");
    EXPECT_EQ(call.props.at("NPUW_ONLINE_ISOLATE").as<std::string>(), "COMPUTE");
    EXPECT_EQ(call.props.at("NPUW_ATTN").as<std::string>(), "STATIC");
    EXPECT_EQ(call.props.at("NPUW_FOLD").as<std::string>(), "NO");
    EXPECT_EQ(call.props.at("NPUW_FUNCALL_ASYNC").as<std::string>(), "NO");
    EXPECT_EQ(call.props.at("NPUW_UNFOLD_IREQS").as<std::string>(), "NO");
}

TEST_F(GQACompiledModelTest, PassesGqaModelThroughWithoutDecomposition) {
    auto model = build_group_query_attention_model();
    ASSERT_GT(count_ops<ov::op::internal::GroupQueryAttention>(model), 0u);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(model, {}, recorder));
    ASSERT_NE(compiled, nullptr);

    // GQA model is passed through unchanged -- the online partitioner handles isolation
    // and folding of GQA blocks via the NPUW_FOLD_ONLY=attn path.
    const auto& call = recorder.only_call();
    EXPECT_GT(count_ops<ov::op::internal::GroupQueryAttention>(call.model), 0u);
}

TEST_F(GQACompiledModelTest, RunsUNQDQBeforeInnerCompilation) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_unqdq_model(ov::element::f16), {{"NPUW_UNQDQ", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "YES");
    EXPECT_EQ(count_ops<ov::op::v1::Multiply>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::FakeQuantize>(call.model), 0u);
}

TEST_F(GQACompiledModelTest, RunsConvToMatmulBeforeInnerCompilation) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_conv_to_matmul_model(), {{"NPUW_UNQDQ", "NO"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "NO");
    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(call.model), 1u);
}

TEST_F(GQACompiledModelTest, RunsConvToMatmulOnDumpedGQAShapeBeforeInnerCompilation) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_dumped_gqa_conv_model(), {{"NPUW_UNQDQ", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "YES");
    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(call.model), 1u);
}

TEST_F(GQACompiledModelTest, RunsConvToMatmulAndUNQDQBeforeInnerCompilation) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(
        compiled = create_compiled_model(build_conv_to_matmul_and_unqdq_model(), {{"NPUW_UNQDQ", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "YES");
    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(call.model), 1u);
    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::FakeQuantize>(call.model), 0u);
}

TEST_F(GQACompiledModelTest, ForwardsPropertyAccessToInnerCompiledModel) {
    auto inner = std::make_shared<PropertyForwardingMockCompiledModel>(build_llm_test_model(), m_plugin);
    auto factory = [inner](const std::shared_ptr<ov::Model>&,
                           const std::shared_ptr<const ov::IPlugin>&,
                           const ov::AnyMap&) -> std::shared_ptr<ov::npuw::ICompiledModel> {
        return inner;
    };

    ov::npuw::GQACompiledModel compiled(build_llm_test_model(), m_plugin, base_props(), factory);

    compiled.set_property({{"NPUW_CWAI", "YES"}});
    EXPECT_EQ(inner->last_set_properties.at("NPUW_CWAI").as<std::string>(), "YES");
    EXPECT_TRUE(compiled.get_property("NPUW_FOLD").as<bool>());
}

TEST(GQACompiledModelSupportsTest, ReturnsTrueForWellFormedGqaTransformerModel) {
    // supports() is the auto-dispatch gate: a well-formed GQA model must also have a
    // dynamic KV-cache to be routed here (see GQACompiledModelSupportsTest.*DynamicShape*).
    auto model = build_full_gqa_transformer_model();
    make_kv_cache_dynamic(model, 2);
    EXPECT_TRUE(ov::npuw::GQACompiledModel::supports(model));
}

TEST(GQACompiledModelSupportsTest, ReturnsFalseForFullyStaticGqaTransformerModel) {
    // identify_case() still classifies a fully static model as a known GQA family, but
    // supports() must not route it here -- this wrapper only bridges a dynamic max_seq_len
    // to the NPU's static-shape requirement.
    auto model = build_full_gqa_transformer_model();  // no dynamic dims anywhere
    EXPECT_NE(ov::npuw::GQACompiledModel::identify_case(model), ov::npuw::GQACompiledModel::Case::Unknown);
    EXPECT_FALSE(ov::npuw::GQACompiledModel::supports(model));
}

TEST(GQACompiledModelSupportsTest, ReturnsFalseForBareGqaOpWithoutSurroundingModelContext) {
    // The op alone (no input_hidden_states/position_ids/KV cache naming) isn't enough
    // evidence -- avoid false positives on unrelated graphs using the op.
    EXPECT_FALSE(ov::npuw::GQACompiledModel::supports(build_group_query_attention_model()));
}

TEST(GQACompiledModelSupportsTest, ReturnsFalseWhenHeadCountsAreInconsistent) {
    // num_heads not evenly divisible by kv_num_heads: not a valid GQA grouping.
    EXPECT_FALSE(ov::npuw::GQACompiledModel::supports(build_full_gqa_transformer_model(5, 2)));
}

TEST(GQACompiledModelSupportsTest, ReturnsFalseForUnconnectedActivationAndPositionIds) {
    // A generic decoder reusing "input_hidden_states"/"position_ids" names for
    // unrelated, disconnected tensors must not be mistaken for a GQA model: ports must
    // actually be wired to the GQA op, not just name-matched.
    auto input_hidden_states = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 16});
    input_hidden_states->set_friendly_name("input_hidden_states");
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 4});
    position_ids->set_friendly_name("position_ids");

    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 1, 16});
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 8, 16});
    past_key->set_friendly_name("past_keys_0");
    auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 8, 16});
    past_value->set_friendly_name("past_values_0");
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto cos_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});
    auto sin_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 8});

    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
        ov::OutputVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache,
                        sin_cache},
        4,
        2,
        0.0f,
        true,
        false);

    auto present_key = std::make_shared<ov::op::v0::Result>(gqa->output(1));
    present_key->set_friendly_name("present_keys_0");
    auto present_value = std::make_shared<ov::op::v0::Result>(gqa->output(2));
    present_value->set_friendly_name("present_values_0");

    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(gqa->output(0)), present_key, present_value};
    ov::ParameterVector params = {input_hidden_states,
                                 position_ids,
                                 query,
                                 key,
                                 value,
                                 past_key,
                                 past_value,
                                 seqlens_k,
                                 total_sequence_length,
                                 cos_cache,
                                 sin_cache};
    auto model = std::make_shared<ov::Model>(results, params, "gqa_disconnected_ports");
    EXPECT_EQ(ov::npuw::GQACompiledModel::identify_case(model), ov::npuw::GQACompiledModel::Case::Unknown);
}

TEST(GQACompiledModelSupportsTest, ReturnsFalseWithoutPositionIds) {
    auto model = build_full_gqa_transformer_model();
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "position_ids") {
            parameter->set_friendly_name("position_ids_removed_for_test");
        }
    }
    EXPECT_FALSE(ov::npuw::GQACompiledModel::supports(model));
}

TEST(GQACompiledModelSupportsTest, ReturnsFalseWithoutPastPresentKvCacheNaming) {
    auto model = build_full_gqa_transformer_model();
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "past_keys_0") {
            parameter->set_friendly_name("layer0_prior_kv_k");
        } else if (parameter->get_friendly_name() == "past_values_0") {
            parameter->set_friendly_name("layer0_prior_kv_v");
        }
    }
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() == "present_keys_0") {
            result->set_friendly_name("layer0_next_kv_k");
        } else if (result->get_friendly_name() == "present_values_0") {
            result->set_friendly_name("layer0_next_kv_v");
        }
    }
    EXPECT_FALSE(ov::npuw::GQACompiledModel::supports(model));
}

TEST(GQACompiledModelSupportsTest, IdentifiesCaseV1ForExplicitPositionIdsModel) {
    // identify_case() classifies purely from structure, regardless of shape-dynamism.
    auto model = build_full_gqa_transformer_model(4, 2, true, PositionSignal::PositionIds);
    EXPECT_EQ(ov::npuw::GQACompiledModel::identify_case(model), ov::npuw::GQACompiledModel::Case::V1);
    // supports() additionally requires a dynamic KV-cache to auto-dispatch.
    make_kv_cache_dynamic(model, 2);
    EXPECT_TRUE(ov::npuw::GQACompiledModel::supports(model));
}

TEST(GQACompiledModelSupportsTest, IdentifiesCaseV0ForSeqLenPairModel) {
    // Wiring: no position_ids input at all -- RoPE
    // position is implied by past_seq_len/total_seq_len instead.
    // identify_case() classifies purely from structure, regardless of shape-dynamism.
    auto model = build_full_gqa_transformer_model(4, 2, true, PositionSignal::SeqLenPair);
    EXPECT_EQ(ov::npuw::GQACompiledModel::identify_case(model), ov::npuw::GQACompiledModel::Case::V0);
    // supports() additionally requires a dynamic KV-cache to auto-dispatch.
    make_kv_cache_dynamic(model, 2);
    EXPECT_TRUE(ov::npuw::GQACompiledModel::supports(model));
}

TEST(GQACompiledModelSupportsTest, IdentifiesCaseV2ForSeqLenPairModelWith2BitWeights) {
    // Same seqlens_k/total_sequence_length wiring as V0, but the model also carries
    // 2-bit group-quantized (u2) weight Parameters -- the distinctive V2 trait.
    auto model = build_full_gqa_transformer_model(4,
                                                  2,
                                                  true,
                                                  PositionSignal::SeqLenPair,
                                                  /*with_2bit_weights=*/true);
    EXPECT_EQ(ov::npuw::GQACompiledModel::identify_case(model), ov::npuw::GQACompiledModel::Case::V2);
    make_kv_cache_dynamic(model, 2);
    EXPECT_TRUE(ov::npuw::GQACompiledModel::supports(model));
}

TEST(GQACompiledModelSupportsTest, IdentifiesCaseUnknownWithoutAnyPositionSignal) {
    auto model = build_full_gqa_transformer_model(4, 2, true, PositionSignal::PositionIds);
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "position_ids") {
            parameter->set_friendly_name("position_ids_removed_for_test");
        }
    }
    EXPECT_EQ(ov::npuw::GQACompiledModel::identify_case(model), ov::npuw::GQACompiledModel::Case::Unknown);
}

TEST(GQACompiledModelDynamicKvCacheTest, ReturnsFalseForFullyStaticModel) {
    auto model = build_full_gqa_transformer_model();
    EXPECT_FALSE(ov::npuw::GQACompiledModel::has_dynamic_max_seq_len(model));
}

TEST(GQACompiledModelDynamicKvCacheTest, DetectsDynamicSeqLenAtAxis2) {
    auto model = build_gqa_model_with_dynamic_kv_cache(2);
    EXPECT_TRUE(ov::npuw::GQACompiledModel::has_dynamic_max_seq_len(model));
}

TEST(GQACompiledModelDynamicKvCacheTest, DetectsDynamicSeqLenAtAxis3) {
    // Mirrors the --transpose_v layout, where the KV-cache dimension moves to the last axis.
    auto model = build_gqa_model_with_dynamic_kv_cache(3);
    EXPECT_TRUE(ov::npuw::GQACompiledModel::has_dynamic_max_seq_len(model));
}

TEST(GQACompiledModelDynamicKvCacheTest, IgnoresDynamicBatchAxis) {
    // Regression test: axis 0 (batch) is never the KV-cache capacity dimension (see
    // GroupQueryAttentionDecomposition, which reads/writes capacity at axis 2 or 3).
    auto model = build_gqa_model_with_dynamic_kv_cache(0);
    EXPECT_FALSE(ov::npuw::GQACompiledModel::has_dynamic_max_seq_len(model));
}

TEST(GQACompiledModelDynamicKvCacheTest, IgnoresDynamicKvHeadsAxis) {
    // Same as above for axis 1 (kv_heads) -- also never the capacity dimension.
    auto model = build_gqa_model_with_dynamic_kv_cache(1);
    EXPECT_FALSE(ov::npuw::GQACompiledModel::has_dynamic_max_seq_len(model));
}

TEST(GQACompiledModelDynamicKvCacheTest, DetectsDynamicAttentionBias) {
    // Speculative-decode-style wiring: the KV-cache itself is static, only the
    // attention mask/bias fed into ATTENTION_BIAS carries a dynamic max_seq_len.
    auto model = build_gqa_model_with_dynamic_attention_bias();
    EXPECT_TRUE(ov::npuw::GQACompiledModel::has_dynamic_max_seq_len(model));
}

TEST(GQACompiledModelDynamicKvCacheTest, ReturnsFalseWhenAttentionBiasIsFullyStatic) {
    auto model = build_gqa_model_with_dynamic_attention_bias();
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "attention_mask") {
            parameter->set_partial_shape(ov::PartialShape{1, 1, 4, 8});
        }
    }
    model->validate_nodes_and_infer_types();
    EXPECT_FALSE(ov::npuw::GQACompiledModel::has_dynamic_max_seq_len(model));
}

TEST_F(GQACompiledModelTest, ReshapesDynamicKvCacheToStaticCapacityAtAxis2) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;
    auto model = build_gqa_model_with_dynamic_kv_cache(2);

    ASSERT_NO_THROW(compiled = create_compiled_model(model, {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_FALSE(call.model->is_dynamic());
    for (const auto& parameter : call.model->get_parameters()) {
        const auto& name = parameter->get_friendly_name();
        if (name == "past_keys_0" || name == "past_values_0") {
            EXPECT_EQ(parameter->get_shape().at(2), 8192u);
        }
    }

    // The outer, user-facing model stays dynamic so the infer request still accepts
    // variable-length KV-cache input.
    EXPECT_TRUE(compiled->inputs().front().get_partial_shape().is_dynamic() ||
               std::any_of(compiled->inputs().begin(), compiled->inputs().end(), [](const auto& input) {
                   return input.get_partial_shape().is_dynamic();
               }));
}

TEST_F(GQACompiledModelTest, ReshapesDynamicKvCacheToStaticCapacityAtAxis3) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;
    auto model = build_gqa_model_with_dynamic_kv_cache(3);

    ASSERT_NO_THROW(compiled = create_compiled_model(model, {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_FALSE(call.model->is_dynamic());
    for (const auto& parameter : call.model->get_parameters()) {
        const auto& name = parameter->get_friendly_name();
        if (name == "past_keys_0" || name == "past_values_0") {
            EXPECT_EQ(parameter->get_shape().at(3), 8192u);
        }
    }
}

TEST_F(GQACompiledModelTest, GetPropertyReturnsEffectiveMaxContextLenInsteadOfThrowing) {
    // Regression test for review comment AlexanderKalistratov#4246978192 on PR
    // openvinotoolkit/openvino#38304: NPUW_LLM_MAX_CONTEXT_LEN is stripped before the
    // inner CompiledModel is created, so forwarding get_property() there throws. It must
    // be answered directly from the wrapper, reflecting the effective capacity used.
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;
    auto model = build_gqa_model_with_dynamic_kv_cache(2);

    ASSERT_NO_THROW(compiled = create_compiled_model(model, {{"NPUW_LLM_MAX_CONTEXT_LEN", "4096"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    ov::Any max_context_len;
    ASSERT_NO_THROW(max_context_len = compiled->get_property("NPUW_LLM_MAX_CONTEXT_LEN"));
    EXPECT_EQ(max_context_len.as<uint32_t>(), 4096u);
}

TEST_F(GQACompiledModelTest, GetPropertyReturnsEffectiveSlidingWindowCapacityInsteadOfThrowing) {
    // Same class of bug as GetPropertyReturnsEffectiveMaxContextLenInsteadOfThrowing:
    // NPUW_LLM_SWA_LEN is also a ROOT-only option stripped before the inner
    // CompiledModel is created, so it must likewise be answered directly from the
    // wrapper instead of being forwarded (and throwing).
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;
    auto model = build_gqa_model_with_dynamic_kv_cache(2);

    ASSERT_NO_THROW(
        compiled = create_compiled_model(model, {{"NPUW_LLM_MAX_CONTEXT_LEN", "16"}, {"NPUW_LLM_SWA_LEN", "8"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    ov::Any sliding_window_len;
    ASSERT_NO_THROW(sliding_window_len = compiled->get_property("NPUW_LLM_SWA_LEN"));
    EXPECT_EQ(sliding_window_len.as<uint32_t>(), 8u);
}

TEST_F(GQACompiledModelTest, ReshapesDynamicAttentionBiasToStaticCapacity) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;
    auto model = build_gqa_model_with_dynamic_attention_bias(/*dynamic_kv_cache=*/true);

    ASSERT_NO_THROW(compiled = create_compiled_model(model, {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_FALSE(call.model->is_dynamic());
    for (const auto& parameter : call.model->get_parameters()) {
        const auto& name = parameter->get_friendly_name();
        if (name == "attention_mask") {
            EXPECT_EQ(parameter->get_shape().at(3), 8192u);
        } else if (name == "past_keys_0" || name == "past_values_0") {
            EXPECT_EQ(parameter->get_shape().at(2), 8192u);
        }
    }
}

TEST_F(GQACompiledModelTest, RejectsDynamicAttentionBiasAxisThatIsNotTheCapacityAxis) {
    // Regression test for review comment AlexanderKalistratov#4246980391 on PR
    // openvinotoolkit/openvino#38304: attention_bias's capacity dimension is always the
    // LAST axis, never query-length. A model leaving axis 2 (query length) dynamic
    // instead must be rejected, not silently pinned.
    auto model = build_gqa_model_with_dynamic_attention_bias(/*dynamic_kv_cache=*/true);
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "attention_mask") {
            // Dynamic axis moved off the last position, which is now statically 8.
            parameter->set_partial_shape(ov::PartialShape{1, 1, ov::Dimension::dynamic(), 8});
        }
    }
    model->validate_nodes_and_infer_types();

    RecordingFactory recorder;
    EXPECT_THROW(create_compiled_model(model, {}, recorder), ov::Exception);
}

namespace {
// A minimal fake "inner" compiled model / infer request pair for exercising
// GQAInferRequest::set_tensor()/get_tensor() end-to-end (unlike RecordingFactory/
// MockSubCompiledModel, whose create_sync_infer_request() returns nullptr). Every port
// gets a real, pre-allocated tensor -- mirrors TestInferRequest in failsafe.cpp.
class FakeInnerSyncInferRequest final : public ov::ISyncInferRequest {
public:
    explicit FakeInnerSyncInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
        : ov::ISyncInferRequest(compiled_model) {
        for (const auto& input : get_inputs()) {
            ov::ISyncInferRequest::set_tensor(
                input,
                ov::get_tensor_impl(ov::Tensor(input.get_element_type(), input.get_shape())));
        }
        for (const auto& output : get_outputs()) {
            ov::ISyncInferRequest::set_tensor(
                output,
                ov::get_tensor_impl(ov::Tensor(output.get_element_type(), output.get_shape())));
        }
    }
    void infer() override {}
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
};

class FakeInnerCompiledModel final : public ov::npuw::ICompiledModel {
public:
    FakeInnerCompiledModel(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::npuw::ICompiledModel(model, plugin) {}
    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return {};
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return std::make_shared<FakeInnerSyncInferRequest>(shared_from_this());
    }
};

ov::npuw::GQACompiledModel::CompiledModelFactory fake_inner_factory() {
    return [](const std::shared_ptr<ov::Model>& model,
             const std::shared_ptr<const ov::IPlugin>& plugin,
             const ov::AnyMap&) -> std::shared_ptr<ov::npuw::ICompiledModel> {
        return std::make_shared<FakeInnerCompiledModel>(model, plugin);
    };
}

// build_gqa_model_with_dynamic_kv_cache()'s past_key/past_value ports are f16; this
// helper fills a real f16 tensor so set_tensor()'s type-check passes.
ov::SoPtr<ov::ITensor> make_f16_kv_cache_tensor(const ov::Shape& shape, float start_val = 0.f) {
    auto tensor = ov::get_tensor_impl(ov::Tensor(ov::element::f16, shape));
    auto* data = reinterpret_cast<ov::float16*>(tensor->data());
    for (size_t i = 0; i < tensor->get_size(); ++i) {
        data[i] = ov::float16(start_val + static_cast<float>(i));
    }
    return tensor;
}
}  // namespace

namespace {
ov::SoPtr<ov::ITensor> make_kv_cache_tensor(const ov::Shape& shape, float start_val = 0.f) {
    auto tensor = ov::get_tensor_impl(ov::Tensor(ov::element::f32, shape));
    auto* data = reinterpret_cast<float*>(tensor->data());
    for (size_t i = 0; i < tensor->get_size(); ++i) {
        data[i] = start_val + static_cast<float>(i);
    }
    return tensor;
}

std::vector<float> to_vec(const ov::SoPtr<ov::ITensor>& t) {
    const auto* data = reinterpret_cast<const float*>(t->data());
    return std::vector<float>(data, data + t->get_size());
}
}  // namespace

TEST_F(GQACompiledModelTest, SetTensorDoesNotCorruptStaleFullCapacityCallerTensorAfterShortTensor) {
    // Regression test: after a port's inner request got aliased onto a full-capacity
    // caller tensor, a later set_tensor() with a *shorter* tensor left the inner request
    // pointing at the stale, orphaned full tensor; the subsequent sync then wrote the
    // new data straight into it, silently corrupting memory the caller might still hold.
    auto model = build_gqa_model_with_dynamic_kv_cache(2);
    auto compiled = std::make_shared<ov::npuw::GQACompiledModel>(
        model,
        m_plugin,
        ov::AnyMap{{"NPUW_GQA", "YES"}, {"NPUW_LLM_MAX_CONTEXT_LEN", "16"}},
        fake_inner_factory());
    ASSERT_NE(compiled, nullptr);

    const uint32_t capacity = 16;
    const size_t num_heads = 2, head_size = 16;
    auto request = compiled->create_infer_request();
    ASSERT_NE(request, nullptr);

    ov::Output<const ov::Node> past_key_port;
    for (const auto& input : compiled->inputs()) {
        if (input.get_node()->get_friendly_name() == "past_keys_0") {
            past_key_port = input;
        }
    }
    ASSERT_TRUE(past_key_port.get_node());

    // 1) Full-capacity caller tensor -- exercises the "exact capacity" aliasing branch.
    auto full_tensor = make_f16_kv_cache_tensor({1, num_heads, capacity, head_size}, /*start_val=*/0.f);
    request->set_tensor(past_key_port, full_tensor);
    // Poison every element so we can detect any stray writes into this (now stale) tensor.
    auto* full_data = reinterpret_cast<ov::float16*>(full_tensor->data());
    for (size_t i = 0; i < full_tensor->get_size(); ++i) {
        full_data[i] = ov::float16(-1.0f);
    }

    // 2) Short caller tensor for the same port -- used to leave the inner request
    // pointing at `full_tensor` while bookkeeping moved on to the short one.
    const size_t short_len = 4;
    auto short_tensor = make_f16_kv_cache_tensor({1, num_heads, short_len, head_size}, /*start_val=*/0.f);
    request->set_tensor(past_key_port, short_tensor);

    ASSERT_NO_THROW(request->infer());

    // The stale full-capacity caller tensor must be left untouched: it's no longer part
    // of the live request, and the caller may still hold a reference to it.
    for (size_t i = 0; i < full_tensor->get_size(); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(full_data[i]), -1.0f)
            << "stale full-capacity caller tensor was corrupted at index " << i;
    }

    // 3) Going back to a full-capacity tensor afterwards must still work (re-alias path).
    auto full_tensor_2 = make_f16_kv_cache_tensor({1, num_heads, capacity, head_size}, /*start_val=*/100.f);
    ASSERT_NO_THROW(request->set_tensor(past_key_port, full_tensor_2));
    ASSERT_NO_THROW(request->infer());
}

TEST_F(GQACompiledModelTest, SetTensorOnDynamicPresentOutputIsHonoredByGetTensorAfterInfer) {
    // Regression test: present_* outputs (not tracked by m_dynamic_kv_cache_axes, which
    // only covers inputs) fell through to the generic "forward as-is" branch, but
    // refresh_present_tensors_locked() still republished its own separately-copied
    // tensor under the same name once past_len < capacity -- diverging from the exact
    // tensor object the caller set.
    auto model = build_gqa_model_with_dynamic_kv_cache(2);
    auto compiled = std::make_shared<ov::npuw::GQACompiledModel>(
        model,
        m_plugin,
        ov::AnyMap{{"NPUW_GQA", "YES"}, {"NPUW_LLM_MAX_CONTEXT_LEN", "16"}},
        fake_inner_factory());
    ASSERT_NE(compiled, nullptr);

    auto request = compiled->create_infer_request();
    ASSERT_NE(request, nullptr);

    ov::Output<const ov::Node> past_key_port, present_key_port;
    for (const auto& input : compiled->inputs()) {
        if (input.get_node()->get_friendly_name() == "past_keys_0") {
            past_key_port = input;
        }
    }
    for (const auto& output : compiled->outputs()) {
        if (output.get_node()->get_friendly_name() == "present_keys_0") {
            present_key_port = output;
        }
    }
    ASSERT_TRUE(past_key_port.get_node());
    ASSERT_TRUE(present_key_port.get_node());

    // A short past tensor (4 of 16 capacity) normally makes
    // refresh_present_tensors_locked() copy the valid prefix of the present output into a
    // fresh buffer.
    auto short_past = make_f16_kv_cache_tensor({1, 2, 4, 16}, /*start_val=*/0.f);
    request->set_tensor(past_key_port, short_past);

    // The caller IO-binds the present output to its own full-capacity buffer, as real
    // apps do.
    auto caller_present = make_f16_kv_cache_tensor({1, 2, 16, 16}, /*start_val=*/0.f);
    ASSERT_NO_THROW(request->set_tensor(present_key_port, caller_present));

    ASSERT_NO_THROW(request->infer());

    // get_tensor() must keep returning the exact same tensor object the caller set --
    // not a separate, internally-copied prefix.
    auto returned = request->get_tensor(present_key_port);
    EXPECT_EQ(returned._ptr.get(), caller_present._ptr.get());
}

TEST_F(GQACompiledModelTest, GetTensorBeforeSetTensorReturnsZeroSizedDynamicAxisNotFullCapacity) {
    // Regression test: get_tensor() on a dynamic-axis input, called before set_tensor(),
    // used to allocate a fallback tensor at full static capacity -- indistinguishable
    // from "fully populated". Other OV plugins instead zero the dynamic dim, giving an
    // unambiguous "nothing written yet" signal.
    auto model = build_gqa_model_with_dynamic_kv_cache(2);
    auto compiled = std::make_shared<ov::npuw::GQACompiledModel>(
        model,
        m_plugin,
        ov::AnyMap{{"NPUW_GQA", "YES"}, {"NPUW_LLM_MAX_CONTEXT_LEN", "16"}},
        fake_inner_factory());
    ASSERT_NE(compiled, nullptr);

    auto request = compiled->create_infer_request();
    ASSERT_NE(request, nullptr);

    ov::Output<const ov::Node> past_key_port;
    for (const auto& input : compiled->inputs()) {
        if (input.get_node()->get_friendly_name() == "past_keys_0") {
            past_key_port = input;
        }
    }
    ASSERT_TRUE(past_key_port.get_node());

    auto tensor = request->get_tensor(past_key_port);
    ASSERT_NE(tensor, nullptr);
    const auto shape = tensor->get_shape();
    ASSERT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[0], 1u);        // N: static
    EXPECT_EQ(shape[1], 2u);        // H: static
    EXPECT_EQ(shape[2], 0u);        // S: dynamic KV-cache axis -- must be 0, not 16
    EXPECT_EQ(shape[3], 16u);       // E: static
}

TEST_F(GQACompiledModelTest, PresentOutputLenReflectsTotalSeqLenEvenWhenPastInputNeverTouched) {
    // Regression test: refresh_present_tensors_locked() only consulted
    // total_sequence_length once a past_key/value tensor had already been looked up
    // (set_tensor()'d or get_tensor()'d by the caller). A caller that queries every
    // output tensor without ever touching the past_key/value input at all (e.g.
    // up-front IO-binding discovery) got back the full, un-copied capacity instead --
    // the same "ambiguous full tensor" ambiguity already fixed for inputs.
    auto model = build_full_gqa_transformer_model(4, 2, /*do_rotary=*/true, PositionSignal::SeqLenPair);
    make_kv_cache_dynamic(model, /*axis=*/2);
    auto compiled = std::make_shared<ov::npuw::GQACompiledModel>(
        model,
        m_plugin,
        ov::AnyMap{{"NPUW_GQA", "YES"}, {"NPUW_LLM_MAX_CONTEXT_LEN", "16"}},
        fake_inner_factory());
    ASSERT_NE(compiled, nullptr);

    auto request = compiled->create_infer_request();
    ASSERT_NE(request, nullptr);

    ov::Output<const ov::Node> total_seq_len_port, present_key_port;
    for (const auto& input : compiled->inputs()) {
        if (input.get_node()->get_friendly_name() == "total_seq_len") {
            total_seq_len_port = input;
        }
    }
    for (const auto& output : compiled->outputs()) {
        if (output.get_node()->get_friendly_name() == "present_keys_0") {
            present_key_port = output;
        }
    }
    ASSERT_TRUE(total_seq_len_port.get_node());
    ASSERT_TRUE(present_key_port.get_node());

    auto total_seq_len_tensor = ov::get_tensor_impl(ov::Tensor(ov::element::i32, ov::Shape{1}));
    *reinterpret_cast<int32_t*>(total_seq_len_tensor->data()) = 5;
    request->set_tensor(total_seq_len_port, total_seq_len_tensor);

    // past_keys_0/past_values_0 are deliberately never touched here.
    ASSERT_NO_THROW(request->infer());

    auto present_tensor = request->get_tensor(present_key_port);
    ASSERT_NE(present_tensor, nullptr);
    EXPECT_EQ(present_tensor->get_shape().at(2), 5u);  // total_seq_len, not the 16-capacity
}


TEST(GQACompiledModelCopyKvCachePrefixTest, CopiesPrefixAlongAxis2LeftAligned) {
    // [N=1, H=2, S, E=3]: src has S1=2, dst has capacity S2=4.
    auto src = make_kv_cache_tensor({1, 2, 2, 3}, 0.f);
    auto dst = make_kv_cache_tensor({1, 2, 4, 3}, 100.f);

    ASSERT_NO_THROW(ov::npuw::GQAInferRequest::copy_kv_cache_prefix(src, dst, /*axis=*/2));

    const auto dst_values = to_vec(dst);
    // Head 0: src rows [0..5] copied into dst's first 2 (of 4) rows; head 1 likewise.
    EXPECT_EQ(std::vector<float>(dst_values.begin(), dst_values.begin() + 6),
              (std::vector<float>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(std::vector<float>(dst_values.begin() + 12, dst_values.begin() + 18),
              (std::vector<float>{6, 7, 8, 9, 10, 11}));
}

TEST(GQACompiledModelCopyKvCachePrefixTest, CopiesPrefixAlongAxis3LeftAligned) {
    // [N=1, H=1, E=2, S] (transpose_v layout): src has S1=2, dst has capacity S2=4.
    auto src = make_kv_cache_tensor({1, 1, 2, 2}, 0.f);
    auto dst = make_kv_cache_tensor({1, 1, 2, 4}, 100.f);

    ASSERT_NO_THROW(ov::npuw::GQAInferRequest::copy_kv_cache_prefix(src, dst, /*axis=*/3));

    const auto dst_values = to_vec(dst);
    // Row 0 (e=0): src[0,1] into dst's first 2 (of 4) slots; row 1 (e=1) likewise.
    EXPECT_EQ(std::vector<float>(dst_values.begin(), dst_values.begin() + 2), (std::vector<float>{0, 1}));
    EXPECT_EQ(std::vector<float>(dst_values.begin() + 4, dst_values.begin() + 6), (std::vector<float>{2, 3}));
}

// Exercises the wire-format round trip that export_model()/import_model() rely on: outer
// ports and the dynamic-axis map must survive a write/read cycle through the same
// ov::npuw::orc::Stream machinery without losing data.
TEST(GQACompiledModelSerializationTest, RoundTripsOuterModelPortsAndDynamicAxisMap) {
    auto model = build_gqa_model_with_dynamic_kv_cache(2);

    std::vector<ov::Output<const ov::Node>> outer_inputs;
    for (const auto& p : model->get_parameters()) {
        outer_inputs.push_back(p->output(0));
    }
    std::vector<ov::Output<const ov::Node>> outer_outputs;
    for (const auto& r : model->get_results()) {
        outer_outputs.push_back(r->output(0));
    }
    const std::unordered_map<std::string, size_t> axis_map{{"past_keys_0", 2u}, {"past_values_0", 2u}};

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    ov::npuw::GQACompiledModel::write_port_list(buffer, outer_inputs);
    ov::npuw::GQACompiledModel::write_port_list(buffer, outer_outputs);
    {
        auto writer = ov::npuw::orc::Stream::writer(buffer);
        writer & axis_map;
    }

    ov::ParameterVector read_parameters = ov::npuw::GQACompiledModel::read_input_port_list(buffer);
    ov::NodeVector read_results = ov::npuw::GQACompiledModel::read_output_port_list(buffer);
    std::unordered_map<std::string, size_t> read_axis_map;
    {
        auto reader = ov::npuw::orc::Stream::reader(buffer);
        reader & read_axis_map;
    }

    ASSERT_EQ(read_parameters.size(), outer_inputs.size());
    ASSERT_EQ(read_results.size(), outer_outputs.size());
    EXPECT_EQ(read_axis_map, axis_map);

    // Positional correspondence matters: map_port_locked() assumes outer_inputs/outputs
    // line up index-by-index with what's read back, with names/types/shapes preserved.
    for (size_t i = 0; i < outer_inputs.size(); ++i) {
        EXPECT_EQ(read_parameters[i]->get_friendly_name(), outer_inputs[i].get_node()->get_friendly_name());
        EXPECT_EQ(read_parameters[i]->get_element_type(), outer_inputs[i].get_element_type());
        EXPECT_EQ(read_parameters[i]->get_partial_shape(), outer_inputs[i].get_partial_shape());
    }
    for (size_t i = 0; i < outer_outputs.size(); ++i) {
        EXPECT_EQ(read_results[i]->get_friendly_name(), outer_outputs[i].get_node()->get_friendly_name());
        EXPECT_EQ(read_results[i]->get_output_element_type(0), outer_outputs[i].get_element_type());
        EXPECT_EQ(read_results[i]->get_output_partial_shape(0), outer_outputs[i].get_partial_shape());
    }

    // The reconstructed vectors must form a valid ov::Model preserving input/output order.
    auto rebuilt_model =
        std::make_shared<ov::Model>(ov::as_output_vector(read_results), read_parameters, "gqa_outer_model");
    ASSERT_EQ(rebuilt_model->inputs().size(), outer_inputs.size());
    for (size_t i = 0; i < outer_inputs.size(); ++i) {
        EXPECT_EQ(rebuilt_model->input(i).get_node()->get_friendly_name(),
                  outer_inputs[i].get_node()->get_friendly_name());
    }

    bool found_dynamic_past_key = false;
    for (const auto& p : read_parameters) {
        if (p->get_friendly_name() == "past_keys_0") {
            found_dynamic_past_key = true;
            EXPECT_TRUE(p->get_partial_shape().is_dynamic());
            ASSERT_GT(p->get_partial_shape().rank().get_length(), 2);
            EXPECT_TRUE(p->get_partial_shape()[2].is_dynamic());
        }
    }
    EXPECT_TRUE(found_dynamic_past_key);
}

// Regression test: ORT/OVEP matches ports via tensor names (Output::get_names()), not
// friendly name. A reconstructed port with dropped tensor names passes friendly-name
// assertions but still breaks ORT-side lookup -- crashed with ACCESS_VIOLATION before.
TEST(GQACompiledModelSerializationTest, RoundTripsTensorNamesSeparatelyFromFriendlyName) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 64});
    param->set_friendly_name("past_keys_0");
    param->output(0).get_tensor().set_names({"past_key_values.0.key"});
    std::vector<ov::Output<const ov::Node>> ports{param->output(0)};

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    ov::npuw::GQACompiledModel::write_port_list(buffer, ports);
    auto read_back = ov::npuw::GQACompiledModel::read_input_port_list(buffer);

    ASSERT_EQ(read_back.size(), 1u);
    EXPECT_EQ(read_back[0]->get_friendly_name(), "past_keys_0");
    EXPECT_EQ(read_back[0]->output(0).get_names(), std::unordered_set<std::string>{"past_key_values.0.key"});
}

// Regression test: import_model()'s outer_model is a dummy carcass with no real
// GroupQueryAttention ops, so find_total_sequence_length_param_name()/
// find_present_to_past_param_names() can't be re-run on it after a cache hit -- they'd
// silently find nothing, and refresh_present_tensors_locked() would then fall back to
// full capacity (returning uncopied/stale KV-cache content to the caller, which crashed
// an ORT consumer expecting the real valid length). export_model() must therefore write
// these two fields out directly, and import_model() must read them back instead of
// recomputing.
TEST(GQACompiledModelSerializationTest, RoundTripsTotalSeqLenParamNameAndPresentToPastMap) {
    const std::optional<std::string> total_seq_len_param_name{"total_seq_len"};
    const std::unordered_map<std::string, std::string> present_to_past{
        {"present_keys_0/sink_port_0", "past_keys_0"},
        {"present_values_0/sink_port_0", "past_values_0"},
    };

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    {
        auto writer = ov::npuw::orc::Stream::writer(buffer);
        auto mutable_name = total_seq_len_param_name;
        auto mutable_map = present_to_past;
        writer & mutable_name;
        writer & mutable_map;
    }

    std::optional<std::string> read_name;
    std::unordered_map<std::string, std::string> read_map;
    {
        auto reader = ov::npuw::orc::Stream::reader(buffer);
        reader & read_name;
        reader & read_map;
    }

    EXPECT_EQ(read_name, total_seq_len_param_name);
    EXPECT_EQ(read_map, present_to_past);
}

// Guards the string-based shape/type round trip for a *bounded* dynamic dimension
// (e.g. "1..8192") and a non-f32 element type, both common for real attention-mask/
// KV-cache ports.
TEST(GQACompiledModelSerializationTest, RoundTripsBoundedDynamicDimensionAndNonF32Type) {
    const ov::PartialShape shape{1, 32, ov::Dimension(1, 8192), 64};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape);
    param->set_friendly_name("attention_bias");
    std::vector<ov::Output<const ov::Node>> ports{param->output(0)};

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    ov::npuw::GQACompiledModel::write_port_list(buffer, ports);
    auto read_back = ov::npuw::GQACompiledModel::read_input_port_list(buffer);

    ASSERT_EQ(read_back.size(), 1u);
    EXPECT_EQ(read_back[0]->get_friendly_name(), "attention_bias");
    EXPECT_EQ(read_back[0]->get_element_type(), ov::element::f16);
    ASSERT_EQ(read_back[0]->get_partial_shape().rank().get_length(), 4);
    EXPECT_EQ(read_back[0]->get_partial_shape()[2], ov::Dimension(1, 8192));
    EXPECT_EQ(read_back[0]->get_partial_shape(), shape);
}

TEST(ReshapeToStaticGQATest, PinsSlidingWindowCacheToItsOwnSmallerCapacityNotModelWideMax) {
    // Regression test for review comment ankitm3k#4230466592 on PR
    // openvinotoolkit/openvino#38304: sliding_window_cache=true means the KV-cache is a
    // fixed-capacity ring buffer, so it must get its own 16-aligned+headroom capacity
    // (6 -> 16 -> 32), independent of the much larger model-wide max_seq_len.
    auto model = build_gqa_model_with_sliding_window_cache(/*axis=*/2);
    ov::npuw::ReshapeToStaticGQA reshape_to_static_gqa(/*max_seq_len=*/8192);
    EXPECT_NO_THROW(reshape_to_static_gqa.run_on_model(model));

    const auto& params = model->get_parameters();
    auto it = std::find_if(params.begin(), params.end(), [](const auto& parameter) {
        return parameter->get_friendly_name() == "past_keys_0";
    });
    ASSERT_NE(it, params.end());
    EXPECT_EQ((*it)->get_partial_shape(), ov::PartialShape({1, 2, 32, 16}));
}

TEST(ReshapeToStaticGQATest, MixedGlobalAndSlidingLayersKeepDistinctCapacities) {
    // The sliding layer's small capacity must not leak onto the global layer's, and vice
    // versa -- both layers share the same model-wide max_seq_len argument, but only the
    // global layer should actually end up pinned to it.
    auto model = build_gqa_model_with_mixed_global_and_sliding_layers();
    ov::npuw::ReshapeToStaticGQA reshape_to_static_gqa(/*max_seq_len=*/8192);
    EXPECT_NO_THROW(reshape_to_static_gqa.run_on_model(model));

    const auto& params = model->get_parameters();
    auto shape_of = [&](const std::string& name) {
        auto it = std::find_if(params.begin(), params.end(), [&](const auto& parameter) {
            return parameter->get_friendly_name() == name;
        });
        OPENVINO_ASSERT(it != params.end(), "parameter '", name, "' not found");
        return (*it)->get_partial_shape();
    };

    EXPECT_EQ(shape_of("past_keys_global"), ov::PartialShape({1, 2, 8192, 16}));
    EXPECT_EQ(shape_of("past_values_global"), ov::PartialShape({1, 2, 8192, 16}));
    EXPECT_EQ(shape_of("past_keys_sliding"), ov::PartialShape({1, 2, 32, 16}));
    EXPECT_EQ(shape_of("past_values_sliding"), ov::PartialShape({1, 2, 32, 16}));
}

TEST(ReshapeToStaticGQATest, UsesRtInfoSlidingWindowLayersAsExactCapacityNotMaxSeqLen) {
    // Models declare windowing via rt_info["framework"]["sliding_window_layers"/
    // "sliding_window_size"] instead of op attributes. The declared size (2048) must be
    // used VERBATIM (exact contract with host eviction logic), while max_seq_len (8192)
    // still applies to the layer not listed in sliding_window_layers.
    auto model = build_gqa_model_with_rt_info_sliding_window_layers(/*declared_sliding_window_size=*/2048);
    ov::npuw::ReshapeToStaticGQA reshape_to_static_gqa(/*max_seq_len=*/8192);
    EXPECT_NO_THROW(reshape_to_static_gqa.run_on_model(model));

    const auto& params = model->get_parameters();
    auto shape_of = [&](const std::string& name) {
        auto it = std::find_if(params.begin(), params.end(), [&](const auto& parameter) {
            return parameter->get_friendly_name() == name;
        });
        OPENVINO_ASSERT(it != params.end(), "parameter '", name, "' not found");
        return (*it)->get_partial_shape();
    };

    EXPECT_EQ(shape_of("past_keys_global"), ov::PartialShape({1, 2, 8192, 16}));
    EXPECT_EQ(shape_of("past_values_global"), ov::PartialShape({1, 2, 8192, 16}));
    EXPECT_EQ(shape_of("past_keys_sliding"), ov::PartialShape({1, 2, 2048, 16}));
    EXPECT_EQ(shape_of("past_values_sliding"), ov::PartialShape({1, 2, 2048, 16}));
}

TEST(ReshapeToStaticGQATest, RtInfoSlidingWindowSizeIsCappedByMaxSeqLen) {
    // A declared sliding_window_size larger than the configured max_seq_len must not
    // blow past the model-wide context budget -- cap it instead of pinning it verbatim.
    auto model = build_gqa_model_with_rt_info_sliding_window_layers(/*declared_sliding_window_size=*/4096);
    ov::npuw::ReshapeToStaticGQA reshape_to_static_gqa(/*max_seq_len=*/1024);
    EXPECT_NO_THROW(reshape_to_static_gqa.run_on_model(model));

    const auto& params = model->get_parameters();
    auto it = std::find_if(params.begin(), params.end(), [](const auto& parameter) {
        return parameter->get_friendly_name() == "past_keys_sliding";
    });
    ASSERT_NE(it, params.end());
    EXPECT_EQ((*it)->get_partial_shape(), ov::PartialShape({1, 2, 1024, 16}));
}

TEST(ReshapeToStaticGQATest, SeqlensNameHintFallbackAppliesConfiguredDefaultCapacity) {
    // Last-resort case: no rt_info, no op attribute -- only a shared SEQLENS_K/
    // TOTAL_SEQUENCE_LENGTH name hinting "sliding" plus a caller-configured default.
    // The sliding layer gets the configured default; the full layer keeps max_seq_len.
    auto model = build_gqa_model_with_seqlens_name_hint(/*name_hint=*/true);
    ov::npuw::ReshapeToStaticGQA reshape_to_static_gqa(/*max_seq_len=*/8192, /*default_sliding_window_capacity=*/2048);
    EXPECT_NO_THROW(reshape_to_static_gqa.run_on_model(model));

    const auto& params = model->get_parameters();
    auto shape_of = [&](const std::string& name) {
        auto it = std::find_if(params.begin(), params.end(), [&](const auto& parameter) {
            return parameter->get_friendly_name() == name;
        });
        OPENVINO_ASSERT(it != params.end(), "parameter '", name, "' not found");
        return (*it)->get_partial_shape();
    };

    EXPECT_EQ(shape_of("past_keys_global"), ov::PartialShape({1, 2, 8192, 16}));
    EXPECT_EQ(shape_of("past_values_global"), ov::PartialShape({1, 2, 8192, 16}));
    EXPECT_EQ(shape_of("past_keys_sliding"), ov::PartialShape({1, 2, 2048, 16}));
    EXPECT_EQ(shape_of("past_values_sliding"), ov::PartialShape({1, 2, 2048, 16}));
}

TEST(ReshapeToStaticGQATest, SeqlensNameHintFallbackIsOptInOnly) {
    // The fallback must do nothing unless the caller configured a default -- never
    // guessed implicitly, to avoid silently reshaping a model that wasn't asked for.
    auto model = build_gqa_model_with_seqlens_name_hint(/*name_hint=*/true);
    ov::npuw::ReshapeToStaticGQA reshape_to_static_gqa(/*max_seq_len=*/8192);
    EXPECT_NO_THROW(reshape_to_static_gqa.run_on_model(model));

    const auto& params = model->get_parameters();
    auto it = std::find_if(params.begin(), params.end(), [](const auto& parameter) {
        return parameter->get_friendly_name() == "past_keys_sliding";
    });
    ASSERT_NE(it, params.end());
    EXPECT_EQ((*it)->get_partial_shape(), ov::PartialShape({1, 2, 8192, 16}));
}

TEST(ReshapeToStaticGQATest, SeqlensNameHintFallbackDoesNothingWithoutNameHint) {
    // No rt_info, no op attribute, no "sliding" name hint -- no signal to act on, so a
    // configured default must not affect either layer.
    auto model = build_gqa_model_with_seqlens_name_hint(/*name_hint=*/false);
    ov::npuw::ReshapeToStaticGQA reshape_to_static_gqa(/*max_seq_len=*/8192, /*default_sliding_window_capacity=*/2048);
    EXPECT_NO_THROW(reshape_to_static_gqa.run_on_model(model));

    const auto& params = model->get_parameters();
    auto shape_of = [&](const std::string& name) {
        auto it = std::find_if(params.begin(), params.end(), [&](const auto& parameter) {
            return parameter->get_friendly_name() == name;
        });
        OPENVINO_ASSERT(it != params.end(), "parameter '", name, "' not found");
        return (*it)->get_partial_shape();
    };

    EXPECT_EQ(shape_of("past_keys_global"), ov::PartialShape({1, 2, 8192, 16}));
    EXPECT_EQ(shape_of("past_keys_sliding"), ov::PartialShape({1, 2, 8192, 16}));
}

TEST(ReshapeToStaticGQATest, AllowsOrdinaryDynamicKvCacheWithoutSlidingWindowCache) {
    // Sanity check: ordinary (sliding_window_cache=false) dynamic KV-cache models must
    // keep reshaping successfully -- the new check must not become overly broad.
    auto model = build_gqa_model_with_dynamic_kv_cache(/*axis=*/2);
    ov::npuw::ReshapeToStaticGQA reshape_to_static_gqa(/*max_seq_len=*/16);
    EXPECT_NO_THROW(reshape_to_static_gqa.run_on_model(model));
}

TEST(GQACompiledModelPresentAxisTest, FindsDynamicPresentOutputsAtAxis2) {
    // past_keys_0/past_values_0's dynamic axis propagates (present = past + current) to
    // present_keys_0/present_values_0 becoming dynamic too.
    auto model = build_gqa_model_with_dynamic_kv_cache(2);
    std::vector<ov::Output<const ov::Node>> outputs;
    for (const auto& result : model->get_results()) {
        std::shared_ptr<const ov::Node> node = result;
        outputs.emplace_back(node, 0);
    }
    auto axes = ov::npuw::GQACompiledModel::find_dynamic_kv_cache_output_axes(outputs);
    ASSERT_EQ(axes.count("present_keys_0"), 1u);
    ASSERT_EQ(axes.count("present_values_0"), 1u);
    EXPECT_EQ(axes.at("present_keys_0"), 2u);
    EXPECT_EQ(axes.at("present_values_0"), 2u);
}

TEST(GQACompiledModelPresentAxisTest, IgnoresFullyStaticPresentOutputs) {
    auto model = build_full_gqa_transformer_model();  // no dynamic dims anywhere
    std::vector<ov::Output<const ov::Node>> outputs;
    for (const auto& result : model->get_results()) {
        std::shared_ptr<const ov::Node> node = result;
        outputs.emplace_back(node, 0);
    }
    auto axes = ov::npuw::GQACompiledModel::find_dynamic_kv_cache_output_axes(outputs);
    EXPECT_TRUE(axes.empty());
}

TEST(GQACompiledModelPresentAxisTest, MatchesSinkPortSuffixedFriendlyName) {
    // Mirrors the ONNX frontend's Result-node convention: appends "/sink_port_0" to the
    // friendly name while the tensor name stays clean.
    auto model = build_gqa_model_with_dynamic_kv_cache(2);
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() == "present_keys_0") {
            result->set_friendly_name("present_keys_0/sink_port_0");
        }
    }
    std::vector<ov::Output<const ov::Node>> outputs;
    for (const auto& result : model->get_results()) {
        std::shared_ptr<const ov::Node> node = result;
        outputs.emplace_back(node, 0);
    }
    auto axes = ov::npuw::GQACompiledModel::find_dynamic_kv_cache_output_axes(outputs);
    ASSERT_EQ(axes.count("present_keys_0/sink_port_0"), 1u);
    EXPECT_EQ(axes.at("present_keys_0/sink_port_0"), 2u);
}

TEST(GQACompiledModelPresentAxisTest, MatchesDotStylePresentNamingScheme) {
    // Regression test: identify_case() accepts dot-style present naming (e.g.
    // "present.0.key"), but find_dynamic_kv_cache_output_axes() used to only match
    // "present_key"/"present_value" -- leaving dot-style present outputs uncopied and
    // get_tensor() returning the full static inner tensor.
    auto model = build_gqa_model_with_dynamic_kv_cache(2);
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() == "present_keys_0") {
            result->set_friendly_name("present.0.key");
        } else if (result->get_friendly_name() == "present_values_0") {
            result->set_friendly_name("present.0.value");
        }
    }
    std::vector<ov::Output<const ov::Node>> outputs;
    for (const auto& result : model->get_results()) {
        std::shared_ptr<const ov::Node> node = result;
        outputs.emplace_back(node, 0);
    }
    auto axes = ov::npuw::GQACompiledModel::find_dynamic_kv_cache_output_axes(outputs);
    ASSERT_EQ(axes.count("present.0.key"), 1u);
    ASSERT_EQ(axes.count("present.0.value"), 1u);
    EXPECT_EQ(axes.at("present.0.key"), 2u);
    EXPECT_EQ(axes.at("present.0.value"), 2u);
}

// find_present_to_past_param_names() derives the present->past mapping from the GQA
// op's own graph connections, not string name-rewriting -- so it works regardless of
// the model's actual naming convention.
TEST(GQACompiledModelPresentToPastParamNamesTest, MapsUnderscoreStyleNames) {
    auto model = build_gqa_model_with_dynamic_kv_cache(2);
    auto map = ov::npuw::GQACompiledModel::find_present_to_past_param_names(model);
    ASSERT_EQ(map.count("present_keys_0"), 1u);
    ASSERT_EQ(map.count("present_values_0"), 1u);
    EXPECT_EQ(map.at("present_keys_0"), "past_keys_0");
    EXPECT_EQ(map.at("present_values_0"), "past_values_0");
}

// Regression test: onnxruntime-genai dot-style naming has present outputs
// "present.0.key"/"present.0.value" but past inputs "past_key_values.0.key"/
// "past_key_values.0.value" -- a naive substring swap would wrongly produce "past.0.key",
// matching no real Parameter and silently defeating the present->past length lookup.
TEST(GQACompiledModelPresentToPastParamNamesTest, MapsDotStyleNamesNotMatchingSubstringSwap) {
    auto model = build_gqa_model_with_dynamic_kv_cache(2);
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "past_keys_0") {
            parameter->set_friendly_name("past_key_values.0.key");
        } else if (parameter->get_friendly_name() == "past_values_0") {
            parameter->set_friendly_name("past_key_values.0.value");
        }
    }
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() == "present_keys_0") {
            result->set_friendly_name("present.0.key");
        } else if (result->get_friendly_name() == "present_values_0") {
            result->set_friendly_name("present.0.value");
        }
    }
    auto map = ov::npuw::GQACompiledModel::find_present_to_past_param_names(model);
    ASSERT_EQ(map.count("present.0.key"), 1u);
    ASSERT_EQ(map.count("present.0.value"), 1u);
    EXPECT_EQ(map.at("present.0.key"), "past_key_values.0.key");
    EXPECT_EQ(map.at("present.0.value"), "past_key_values.0.value");
}

}  // namespace

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Frontend operations for block-sparse attention (QSAIndexer, SparseAttention) and for hashed n-gram embedding rows.
// They have no ggml counterpart: a native builder emits them with explicit parameters. Activations use the builder
// layout, where a [width, tokens] ggml value is the OpenVINO [batch, 1, tokens, width].

#include "openvino/op/sparse_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/ngram_hash_function.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

using namespace ov::op;

namespace {

std::shared_ptr<v0::Constant> i64s(const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, {values.size()}, values);
}

ov::Output<ov::Node> as_type(const ov::Output<ov::Node>& value, const ov::element::Type& type) {
    return value.get_element_type() == type ? value : std::make_shared<v0::Convert>(value, type)->output(0);
}

// [batch, 1, tokens, width] -> [batch, tokens, width]
ov::Output<ov::Node> token_rows(const ov::Output<ov::Node>& value, int64_t width) {
    return std::make_shared<v1::Reshape>(value, i64s({0, -1, width}), true);
}

ov::Output<ov::Node> flat_vector(const ov::Output<ov::Node>& value, const ov::element::Type& type) {
    return as_type(std::make_shared<v1::Reshape>(value, i64s({-1}), false), type);
}

int64_t static_last_dim(const NodeContext& context, size_t input, const char* what) {
    const auto shape = context.get_input(static_cast<int>(input)).get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(
        shape.rank().is_static() && shape.rank().get_length() > 0 && shape[shape.rank().get_length() - 1].is_static(),
        what,
        " requires a static last dimension");
    return shape[shape.rank().get_length() - 1].get_length();
}

}  // namespace

// cos and sin of every token, [batch, tokens, n_dims], in the rotate-half layout (the n_dims/2
// frequencies twice), for operations that take explicit RoPE tables.
OutputVector translate_rope_cache(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    const auto rope_config = context.get_attribute<RopeConfig>("rope_config");
    FRONT_END_OP_CONVERSION_CHECK(rope_config.n_dims > 0 && rope_config.n_dims % 2 == 0,
                                  "RoPE cache requires a positive even n_dims");
    const int64_t half = rope_config.n_dims / 2;
    auto sin_cos =
        make_sin_cos(rope_config, context.get_input(0).get_node_shared_ptr(), nullptr, rope_config.is_imrope);
    const auto table = [&](const ov::Output<ov::Node>& value) -> ov::Output<ov::Node> {
        auto rows = std::make_shared<v1::Reshape>(value, i64s({0, -1, half}), true);
        return std::make_shared<v0::Concat>(ov::OutputVector{rows, rows}, -1);
    };
    return rename_outputs_with_suffix({table(sin_cos.second), table(sin_cos.first)}, context.get_name());
}

// Number of keys visible to the last query of every sequence of an additive attention mask
// [batch, 1, queries, keys]: the valid key count of a causal, right-aligned (left-padded) batch.
OutputVector translate_attention_key_length(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto mask = context.get_input(0);
    auto last_row = std::make_shared<v8::Slice>(mask,
                                                i64s({-1}),
                                                i64s({std::numeric_limits<int64_t>::max()}),
                                                i64s({1}),
                                                i64s({2}));
    auto zero = v0::Constant::create(mask.get_element_type(), {}, {0});
    auto visible = std::make_shared<v0::Convert>(std::make_shared<v1::Equal>(last_row, zero), ov::element::i32);
    auto count = std::make_shared<v1::ReduceSum>(visible, i64s({1, 2, 3}), false);
    return rename_outputs_with_suffix({count}, context.get_name());
}

// QSAIndexer: block selection for Qwen Sparse Attention with the index query RMSNorm and rotate-half
// RoPE (first n_dims channels) applied inside the operation. It takes the block-key cache as is and
// returns it grown to the complete blocks of the longest row.
// Inputs: index query [batch, 1, tokens, heads * Di], q_norm_weight, raw index keys [batch, 1, tokens, Di],
// k_norm_weight, cos, sin [batch or 1, tokens, n_dims], key_length [batch], and the states
// block keys [batch, blocks, Di], incomplete raw keys [batch, block - 1, Di], incomplete RoPE row
// [batch, 2 * n_dims]. Outputs: block indices, block count and the updated states.
OutputVector translate_qsa_indexer(const NodeContext& context) {
    num_inputs_check(context, 10, 10);
    using Indexer = ov::op::internal::QSAIndexer;
    const auto heads = context.get_attribute<int64_t>("heads");
    const auto block = context.get_attribute<int64_t>("key_block_size");
    const auto budget = context.get_attribute<int64_t>("token_budget");
    const auto eps = context.get_attribute<float>("eps");
    const int64_t width = static_last_dim(context, 0, "QSA index query");
    const int64_t head_dim = static_last_dim(context, 2, "QSA raw index keys");
    const int64_t rotary = static_last_dim(context, 4, "QSA index RoPE table");
    FRONT_END_OP_CONVERSION_CHECK(heads > 0 && width == heads * head_dim,
                                  "QSA index query width must split into heads");
    FRONT_END_OP_CONVERSION_CHECK(rotary > 0 && rotary % 2 == 0 && rotary <= head_dim,
                                  "QSA index RoPE width must be even and within the head");
    FRONT_END_OP_CONVERSION_CHECK(block > 0 && budget > 0 && budget % block == 0,
                                  "QSA token budget must be a positive multiple of the key block size");

    auto raw_keys = std::make_shared<v1::Reshape>(context.get_input(2), i64s({0, -1, 1, head_dim}), true);
    const auto type = raw_keys->get_output_element_type(0);
    auto query =
        std::make_shared<v1::Reshape>(as_type(context.get_input(0), type), i64s({0, -1, heads, head_dim}), true);
    // The operation takes cos / sin per row; the shared tables may have a single row.
    auto rows_shape = std::make_shared<v0::Concat>(
        ov::OutputVector{std::make_shared<v8::Slice>(std::make_shared<v3::ShapeOf>(raw_keys, ov::element::i64),
                                                     i64s({0}),
                                                     i64s({2}),
                                                     i64s({1})),
                         i64s({rotary})},
        0);
    const auto per_row = [&](const ov::Output<ov::Node>& rows) -> ov::Output<ov::Node> {
        return std::make_shared<v3::Broadcast>(as_type(rows, type), rows_shape);
    };
    const auto with_head = [&](const ov::Output<ov::Node>& state) -> ov::Output<ov::Node> {
        return std::make_shared<v0::Unsqueeze>(as_type(state, type), i64s({1}));
    };

    Indexer::Config config;
    config.key_block_size = static_cast<size_t>(block);
    config.block_topk = static_cast<size_t>(budget / block);
    config.num_selection_heads = 1;
    config.indexer_rotary_dim = static_cast<size_t>(rotary);
    config.indexer_head_dim = static_cast<size_t>(head_dim);
    config.epsilon = eps;
    config.scale = Indexer::effective_scale(config);
    config.query_norm = true;
    config.query_rope = true;
    ov::OutputVector inputs(Indexer::InputIdx::COUNT);
    inputs[Indexer::InputIdx::INDEX_QUERY] = query;
    inputs[Indexer::InputIdx::RAW_INDEX_KEYS] = raw_keys;
    inputs[Indexer::InputIdx::ROPE_COS] = per_row(context.get_input(4));
    inputs[Indexer::InputIdx::ROPE_SIN] = per_row(context.get_input(5));
    inputs[Indexer::InputIdx::KEY_LENGTH] = as_type(context.get_input(6), ov::element::i32);
    inputs[Indexer::InputIdx::Q_NORM_WEIGHT] = flat_vector(context.get_input(1), type);
    inputs[Indexer::InputIdx::K_NORM_WEIGHT] = flat_vector(context.get_input(3), type);
    inputs[Indexer::InputIdx::BLOCK_INDEX_KEYS_CACHE] = with_head(context.get_input(7));
    inputs[Indexer::InputIdx::PAST_INCOMPLETE_RAW_K] = with_head(context.get_input(8));
    inputs[Indexer::InputIdx::PAST_INCOMPLETE_ROPE] = as_type(context.get_input(9), type);
    auto indexer = std::make_shared<Indexer>(inputs, config);

    const auto without_head = [&](size_t output) -> ov::Output<ov::Node> {
        return std::make_shared<v0::Squeeze>(indexer->output(output), i64s({1}));
    };
    return rename_outputs_with_suffix({indexer->output(Indexer::OutputIdx::SELECTED_BLOCK_INDICES),
                                       indexer->output(Indexer::OutputIdx::SELECTED_BLOCK_COUNT),
                                       without_head(Indexer::OutputIdx::BLOCK_INDEX_KEYS_CACHE),
                                       without_head(Indexer::OutputIdx::PAST_INCOMPLETE_RAW_K),
                                       indexer->output(Indexer::OutputIdx::PAST_INCOMPLETE_ROPE)},
                                      context.get_name());
}

// SparseAttention(query [batch, tokens, heads, E], key [batch, kv_heads, keys, E],
// value [batch, kv_heads, keys, Ev], block_indices, block_count, key_length) with the default
// 1/sqrt(E) scale unless "scale" differs. The output keeps the query layout [batch, tokens, heads, Ev].
OutputVector translate_sparse_attention(const NodeContext& context) {
    num_inputs_check(context, 6, 6);
    const auto block = context.get_attribute<int64_t>("key_block_size");
    auto query = std::make_shared<v1::Transpose>(context.get_input(0), i64s({0, 2, 1, 3}));
    const auto type = query->get_output_element_type(0);
    ov::OutputVector inputs{query,
                            as_type(context.get_input(1), type),
                            as_type(context.get_input(2), type),
                            context.get_input(3),
                            context.get_input(4),
                            context.get_input(5)};
    const auto head_size = query->get_output_partial_shape(0)[3];
    const auto scale = context.get_attribute<float>("scale", 0.0f);
    if (scale != 0.0f) {
        FRONT_END_OP_CONVERSION_CHECK(head_size.is_static(), "Sparse attention scale requires a static head size");
        const float default_scale = 1.0f / std::sqrt(static_cast<float>(head_size.get_length()));
        if (std::abs(scale - default_scale) > 1e-6f * default_scale)
            inputs.push_back(v0::Constant::create(type, {}, {scale}));
    }
    auto attention = std::make_shared<v17::SparseAttention>(inputs, block, 1, true);
    auto output = std::make_shared<v1::Transpose>(attention, i64s({0, 2, 1, 3}));
    return rename_outputs_with_suffix({output}, context.get_name());
}

// Hashed n-gram rows: for the order-n context of every token (the token and its n-1
// predecessors, EOS before the sequence start and across an EOS),
//   row[h] = (XOR_k back(k) * multipliers[k]) % head_vocab_sizes[h] + head_offsets[h]
// with unsigned 64-bit products. Inputs: tokens [.., batch, tokens] and the history of the
// previous n-1 tokens [batch, n-1], stored as token - "history_offset" so a zero state is all EOS.
// Outputs: rows [1, 1, batch, tokens * heads] (head fastest) and the updated history.
// NGramHashFunction keeps the 64-bit arithmetic inside the operation; its signed floor modulo equals
// the unsigned one because every product of a token below "token_limit" stays below 2^63.
OutputVector translate_ngram_hash(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    const auto ngram = context.get_attribute<int64_t>("ngram_size");
    const auto per_ngram = context.get_attribute<int64_t>("heads_per_ngram");
    const auto multipliers = context.get_attribute<std::vector<int64_t>>("multipliers");
    const auto vocab_sizes = context.get_attribute<std::vector<int64_t>>("head_vocab_sizes");
    const auto offsets = context.get_attribute<std::vector<int64_t>>("head_offsets");
    const auto eos = context.get_attribute<int64_t>("eos_token_id");
    const auto history_offset = context.get_attribute<int64_t>("history_offset", 0);
    const int64_t context_tokens = ngram - 1;
    const auto heads = static_cast<size_t>(context_tokens * per_ngram);
    FRONT_END_OP_CONVERSION_CHECK(ngram >= 2 && per_ngram > 0 && multipliers.size() >= static_cast<size_t>(ngram) &&
                                      vocab_sizes.size() >= heads && offsets.size() >= heads,
                                  "Invalid n-gram hash configuration");
    const auto token_limit = context.get_attribute<int64_t>("token_limit");
    FRONT_END_OP_CONVERSION_CHECK(eos >= 0 && eos < token_limit && token_limit <= (int64_t{1} << 31),
                                  "N-gram tokens and EOS must be non-negative int32 values below token_limit");
    for (size_t h = 0; h < heads; ++h) {
        FRONT_END_OP_CONVERSION_CHECK(
            vocab_sizes[h] > 0 && offsets[h] >= 0 && offsets[h] + vocab_sizes[h] <= (int64_t{1} << 31),
            "N-gram head ranges must fit int32 rows");
    }
    for (int64_t k = 0; k < ngram; ++k) {
        const auto multiplier = static_cast<uint64_t>(multipliers[k]);
        FRONT_END_OP_CONVERSION_CHECK(
            multiplier < (uint64_t{1} << 63) &&
                (multiplier == 0 || static_cast<uint64_t>(token_limit - 1) <= ((uint64_t{1} << 63) - 1) / multiplier),
            "N-gram hash products must stay below 2^63");
    }
    const auto i32 = [](int64_t value) {
        return v0::Constant::create(ov::element::i32, {}, {value});
    };

    auto tokens_in = as_type(context.get_input(0), ov::element::i32);
    auto shape = std::make_shared<v3::ShapeOf>(tokens_in, ov::element::i64);
    auto length = std::make_shared<v8::Gather>(shape, i64s({-1}), v0::Constant::create(ov::element::i64, {}, {0}));
    auto tokens = std::make_shared<v1::Reshape>(tokens_in,
                                                std::make_shared<v0::Concat>(ov::OutputVector{i64s({-1}), length}, 0),
                                                false);
    // Padded positions must read as EOS so they do not enter the n-gram of the first real token.
    tokens->get_rt_info()["gguf.padding_token_id"] = eos;
    ov::Output<ov::Node> history = as_type(context.get_input(1), ov::element::i32);
    history = std::make_shared<v1::Add>(history, i32(history_offset));
    // The absolute history; GenAI adaptation turns it into an explicit input of a lookup model.
    history.get_node_shared_ptr()->get_rt_info()["gguf.token_history_offset"] = history_offset;
    auto sequence = std::make_shared<v0::Concat>(ov::OutputVector{history, tokens}, 1);
    const auto shifted = [&](int64_t k) -> ov::Output<ov::Node> {
        auto stop = std::make_shared<v1::Add>(length, i64s({context_tokens - k}));
        return std::make_shared<v8::Slice>(sequence, i64s({context_tokens - k}), stop, i64s({1}), i64s({1}));
    };
    auto eos_c = i32(eos);
    std::vector<ov::Output<ov::Node>> back{shifted(0)};
    ov::Output<ov::Node> valid;
    for (int64_t k = 1; k <= context_tokens; ++k) {
        auto token = shifted(k);
        back.push_back(valid.get_node() ? std::make_shared<v1::Select>(valid, token, eos_c)->output(0) : token);
        auto not_eos = std::make_shared<v1::NotEqual>(token, eos_c);
        valid = valid.get_node() ? std::make_shared<v1::LogicalAnd>(valid, not_eos)->output(0) : not_eos->output(0);
    }
    // Windows [batch, tokens, n], oldest first.
    ov::OutputVector window;
    for (int64_t k = context_tokens; k >= 0; --k)
        window.push_back(std::make_shared<v0::Unsqueeze>(back[k], i64s({-1})));
    auto row_ids = std::make_shared<ov::op::internal::NGramHashFunction>(
        std::make_shared<v0::Concat>(window, -1),
        std::vector<int64_t>(static_cast<size_t>(context_tokens), per_ngram),
        std::vector<int64_t>(multipliers.begin(), multipliers.begin() + ngram),
        std::vector<int64_t>(vocab_sizes.begin(), vocab_sizes.begin() + heads),
        std::vector<int64_t>(offsets.begin(), offsets.begin() + heads));
    auto batch = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(tokens, ov::element::i64),
                                              i64s({0}),
                                              v0::Constant::create(ov::element::i64, {}, {0}));
    auto flat_rows = std::make_shared<v1::Reshape>(
        row_ids,
        std::make_shared<v0::Concat>(ov::OutputVector{i64s({1, 1}), batch, i64s({-1})}, 0),
        false);
    flat_rows->get_rt_info()["gguf.token_indices"] = static_cast<int64_t>(heads);
    ov::Output<ov::Node> new_history = std::make_shared<v8::Slice>(sequence,
                                                                   i64s({-context_tokens}),
                                                                   i64s({std::numeric_limits<int64_t>::max()}),
                                                                   i64s({1}),
                                                                   i64s({1}));
    if (history_offset != 0)
        new_history = std::make_shared<v1::Subtract>(new_history, i32(history_offset));
    new_history = as_type(new_history, context.get_input(1).get_element_type());
    return rename_outputs_with_suffix({flat_rows, new_history}, context.get_name());
}

}  // namespace ov::frontend::gguf::op

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Frontend operations for block-sparse attention (QSAPrepare, SparseAttentionIndexer,
// SparseAttention) and for hashed n-gram embedding rows. They have no ggml counterpart: a native
// builder emits them with explicit parameters. Activations use the builder layout, where a
// [width, tokens] ggml value is the OpenVINO [batch, 1, tokens, width].

#include "openvino/op/sparse_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/decompositions/rms_norm.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/node_registry.hpp"
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

// QSAPrepare(raw_index_keys, past_block_index_keys, key_length, cos, sin, k_norm_weight,
//            past_incomplete_raw_keys, past_incomplete_rope).
// raw_index_keys is [batch, 1, tokens, Di]; the states use the operation layout.
OutputVector translate_qsa_prepare(const NodeContext& context) {
    num_inputs_check(context, 8, 8);
    const auto block = context.get_attribute<int64_t>("key_block_size");
    const auto eps = context.get_attribute<float>("eps");
    const int64_t width = static_last_dim(context, 0, "QSAPrepare raw index keys");
    auto raw = token_rows(context.get_input(0), width);
    const auto type = raw.get_element_type();
    auto prepare = std::make_shared<v17::QSAPrepare>(ov::OutputVector{raw,
                                                                      as_type(context.get_input(1), type),
                                                                      context.get_input(2),
                                                                      as_type(context.get_input(3), type),
                                                                      as_type(context.get_input(4), type),
                                                                      flat_vector(context.get_input(5), type),
                                                                      as_type(context.get_input(6), type),
                                                                      as_type(context.get_input(7), type)},
                                                     block,
                                                     eps);
    return rename_outputs_with_suffix(prepare->outputs(), context.get_name());
}

// SparseAttentionIndexer over an index query projection [batch, 1, tokens, heads * Di] that is
// RMS-normalized per head and rotated with rotate-half RoPE on its first n_dims channels.
// The query rotates with Unsqueeze(axis 1) of the tables given to QSAPrepare, the form fused
// indexers recognize as a shared rotation.
// Inputs: index query, q_norm_weight, cos, sin, block_index_keys, key_length.
OutputVector translate_sparse_attention_indexer(const NodeContext& context) {
    num_inputs_check(context, 6, 6);
    const auto heads = context.get_attribute<int64_t>("heads");
    const auto block = context.get_attribute<int64_t>("key_block_size");
    const auto budget = context.get_attribute<int64_t>("token_budget");
    const auto eps = context.get_attribute<float>("eps");
    const int64_t width = static_last_dim(context, 0, "Sparse attention index query");
    FRONT_END_OP_CONVERSION_CHECK(heads > 0 && width % heads == 0, "Index query width must split into heads");
    const int64_t head_dim = width / heads;
    const int64_t rotary = static_last_dim(context, 2, "Sparse attention index RoPE table");
    FRONT_END_OP_CONVERSION_CHECK(rotary > 0 && rotary % 2 == 0 && rotary <= head_dim,
                                  "Index RoPE width must be even and within the head");

    auto query = std::make_shared<v1::Reshape>(context.get_input(0), i64s({0, -1, heads, head_dim}), true);
    const auto type = query->get_output_element_type(0);
    ov::pass::NodeRegistry registry;
    auto normalized = ov::decomposition::rms_norm(registry,
                                                  query,
                                                  i64s({-1}),
                                                  v0::Constant::create(type, {}, {eps}),
                                                  flat_vector(context.get_input(1), type));
    ov::Output<ov::Node> x = std::make_shared<v1::Transpose>(normalized, i64s({0, 2, 1, 3}));

    auto cos = std::make_shared<v0::Unsqueeze>(as_type(context.get_input(2), type), i64s({1}));
    auto sin = std::make_shared<v0::Unsqueeze>(as_type(context.get_input(3), type), i64s({1}));
    // Split on the explicit last axis: the RoPE fusions match this form before constant normalization.
    auto last_axis = v0::Constant::create(ov::element::i64, {}, {3});
    ov::Output<ov::Node> rotated = x, passed;
    if (rotary < head_dim) {
        auto split = std::make_shared<v1::VariadicSplit>(x, last_axis, i64s({rotary, head_dim - rotary}));
        rotated = split->output(0);
        passed = split->output(1);
    }
    auto halves = std::make_shared<v1::VariadicSplit>(rotated, last_axis, i64s({rotary / 2, rotary / 2}));
    auto negated = std::make_shared<v1::Multiply>(halves->output(1), v0::Constant::create(type, {}, {-1.0f}));
    auto rotate_half = std::make_shared<v0::Concat>(ov::OutputVector{negated, halves->output(0)}, -1);
    ov::Output<ov::Node> roped = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(rotated, cos),
                                                           std::make_shared<v1::Multiply>(rotate_half, sin));
    if (passed.get_node())
        roped = std::make_shared<v0::Concat>(ov::OutputVector{roped, passed}, -1);

    auto indexer = std::make_shared<v17::SparseAttentionIndexer>(roped,
                                                                 as_type(context.get_input(4), type),
                                                                 context.get_input(5),
                                                                 block,
                                                                 budget);
    return rename_outputs_with_suffix(indexer->outputs(), context.get_name());
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
// with exact unsigned 64-bit products. Inputs: tokens [.., batch, tokens] and the history of the
// previous n-1 tokens [batch, n-1], stored as token - "history_offset" so a zero state is all EOS.
// Outputs: rows [1, 1, batch, tokens * heads] (head fastest) and the updated history.
// Plugins run i64 arithmetic in i32 and i32 division through f32, so the hash uses only exact i32
// operations: little-endian limbs, limb-wise products with carries, limb-wise XOR and Horner
// remainders with conditional subtractions.
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
    int64_t max_vocab = 0;
    for (size_t h = 0; h < heads; ++h) {
        FRONT_END_OP_CONVERSION_CHECK(vocab_sizes[h] > 0 && vocab_sizes[h] < (int64_t{1} << 30) && offsets[h] >= 0 &&
                                          offsets[h] + vocab_sizes[h] < (int64_t{1} << 31),
                                      "N-gram head ranges must fit int32 rows");
        max_vocab = std::max(max_vocab, vocab_sizes[h]);
    }
    FRONT_END_OP_CONVERSION_CHECK(eos >= 0 && eos < (int64_t{1} << 31), "N-gram EOS must be a non-negative int32");
    int bits = 0;
    while ((int64_t{1} << bits) <= max_vocab)
        ++bits;
    const int limb_bits = std::min(11, 31 - bits);
    FRONT_END_OP_CONVERSION_CHECK(limb_bits > 0, "N-gram head vocabulary is too large");
    const int64_t limb_base = int64_t{1} << limb_bits;
    const int limbs = (64 + limb_bits - 1) / limb_bits;
    const int token_limbs = (31 + limb_bits - 1) / limb_bits;
    const auto i32 = [](int64_t value) {
        return v0::Constant::create(ov::element::i32, {}, {value});
    };
    const auto low = [&](const ov::Output<ov::Node>& x) -> ov::Output<ov::Node> {
        return std::make_shared<v13::BitwiseAnd>(x, i32(limb_base - 1));
    };
    const auto high = [&](const ov::Output<ov::Node>& x, int64_t shift) -> ov::Output<ov::Node> {
        return std::make_shared<v15::BitwiseRightShift>(x, i32(shift));
    };
    const auto reduce_mod = [&](ov::Output<ov::Node> x, int64_t modulus) {
        for (int j = limb_bits - 1; j >= 0; --j) {
            auto step = i32(modulus << j);
            auto diff = std::make_shared<v1::Subtract>(x, step);
            auto negative = high(diff, 31);
            x = std::make_shared<v1::Add>(diff, std::make_shared<v13::BitwiseAnd>(step, negative));
        }
        return x;
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
    const auto product_limbs = [&](const ov::Output<ov::Node>& token, uint64_t multiplier) {
        std::vector<ov::Output<ov::Node>> token_parts;
        for (int j = 0; j < token_limbs; ++j)
            token_parts.push_back(low(j ? high(token, limb_bits * j) : token));
        std::vector<ov::Output<ov::Node>> result;
        ov::Output<ov::Node> carry;
        for (int c = 0; c < limbs; ++c) {
            ov::Output<ov::Node> column = carry;
            for (int j = 0; j <= std::min(c, token_limbs - 1); ++j) {
                if (limb_bits * (c - j) >= 64)
                    continue;
                const auto part = static_cast<int64_t>((multiplier >> (limb_bits * (c - j))) & (limb_base - 1));
                if (part == 0)
                    continue;
                ov::Output<ov::Node> term = std::make_shared<v1::Multiply>(token_parts[j], i32(part));
                column = column.get_node() ? std::make_shared<v1::Add>(column, term)->output(0) : term;
            }
            if (!column.get_node())
                column = std::make_shared<v1::Multiply>(token, i32(0));
            result.push_back(low(column));
            carry = high(column, limb_bits);
        }
        return result;
    };
    ov::OutputVector rows;
    auto mix = product_limbs(back[0], static_cast<uint64_t>(multipliers[0]));
    for (int64_t order = 2; order <= ngram; ++order) {
        auto term = product_limbs(back[order - 1], static_cast<uint64_t>(multipliers[order - 1]));
        for (size_t i = 0; i < mix.size(); ++i)
            mix[i] = std::make_shared<v13::BitwiseXor>(mix[i], term[i]);
        for (int64_t e = 0; e < per_ngram; ++e) {
            const auto head = static_cast<size_t>((order - 2) * per_ngram + e);
            ov::Output<ov::Node> remainder;
            for (auto it = mix.rbegin(); it != mix.rend(); ++it) {
                remainder =
                    remainder.get_node()
                        ? std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(remainder, i32(limb_base)), *it)
                              ->output(0)
                        : *it;
                remainder = reduce_mod(remainder, vocab_sizes[head]);
            }
            auto row = std::make_shared<v1::Add>(remainder, i32(offsets[head]));
            rows.push_back(std::make_shared<v0::Unsqueeze>(row, i64s({-1})));
        }
    }
    auto row_ids = std::make_shared<v0::Concat>(rows, -1);
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

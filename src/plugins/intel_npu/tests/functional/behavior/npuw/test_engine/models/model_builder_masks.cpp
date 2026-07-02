// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder_masks.hpp"

#include "model_builder_internal.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

ov::Output<ov::Node> make_padding_mask(const ov::Output<ov::Node>& attention_mask_output,
                                       ov::element::Type prec) {
    auto mask_float = std::make_shared<ov::opset11::Convert>(attention_mask_output, prec);
    mask_float->set_friendly_name("model.mask_convert");

    auto one_const = ov::opset11::Constant::create(prec, ov::Shape{}, {1.0f});
    auto inv_mask = std::make_shared<ov::opset11::Subtract>(one_const, mask_float);
    inv_mask->set_friendly_name("model.mask_invert");

    auto neg_inf = ov::opset11::Constant::create(prec, ov::Shape{}, {kAttentionMaskPadding});
    auto padding_mask = std::make_shared<ov::opset11::Multiply>(inv_mask, neg_inf);
    padding_mask->set_friendly_name("model.padding_mask");

    auto pad_shape = ov::opset11::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 1, -1});
    auto padding_4d = std::make_shared<ov::opset11::Reshape>(padding_mask, pad_shape, true);
    padding_4d->set_friendly_name("model.padding_mask_4d");

    return padding_4d->output(0);
}

CausalBool make_causal_bool(const ov::Output<ov::Node>& seq_source,
                            const ov::Output<ov::Node>& attention_mask,
                            const std::string& prefix) {
    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(seq_source, ov::element::i64);
    ids_shape->set_friendly_name(prefix + "ids_shape");

    auto mask_shape = std::make_shared<ov::opset11::ShapeOf>(attention_mask, ov::element::i64);
    mask_shape->set_friendly_name(prefix + "mask_shape");

    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});

    auto seq_len = std::make_shared<ov::opset11::Gather>(ids_shape, idx1, axis0);
    seq_len->set_friendly_name(prefix + "seq_len");

    auto total_seq = std::make_shared<ov::opset11::Gather>(mask_shape, idx1, axis0);
    total_seq->set_friendly_name(prefix + "total_seq");

    auto offset = std::make_shared<ov::opset11::Subtract>(total_seq, seq_len);
    offset->set_friendly_name(prefix + "offset");

    auto range_start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto range_step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});

    auto kv_range = std::make_shared<ov::op::v4::Range>(range_start, total_seq, range_step, ov::element::i64);
    kv_range->set_friendly_name(prefix + "kv_range");

    auto q_range = std::make_shared<ov::op::v4::Range>(range_start, seq_len, range_step, ov::element::i64);
    q_range->set_friendly_name(prefix + "q_range");

    auto q_abs = std::make_shared<ov::opset11::Add>(q_range, offset);
    q_abs->set_friendly_name(prefix + "q_abs_positions");

    auto axis_last = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto axis_first = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});

    auto q_col = std::make_shared<ov::opset11::Unsqueeze>(q_abs, axis_last);
    q_col->set_friendly_name(prefix + "q_col");

    auto kv_row = std::make_shared<ov::opset11::Unsqueeze>(kv_range, axis_first);
    kv_row->set_friendly_name(prefix + "kv_row");

    auto causal_bool = std::make_shared<ov::op::v1::LessEqual>(kv_row, q_col);
    causal_bool->set_friendly_name(prefix + "causal_bool");

    return {causal_bool->output(0), q_col->output(0), kv_row->output(0)};
}

ov::Output<ov::Node> make_causal_mask(const ov::Output<ov::Node>& input_ids_output,
                                      const ov::Output<ov::Node>& attention_mask_output,
                                      ov::element::Type prec) {
    auto padding_4d = make_padding_mask(attention_mask_output, prec);
    auto cb = make_causal_bool(input_ids_output, attention_mask_output, "model.");

    auto select_true = ov::opset11::Constant::create(prec, ov::Shape{}, {0.0f});
    auto select_false = ov::opset11::Constant::create(prec, ov::Shape{}, {kAttentionMaskPadding});

    auto causal_float = std::make_shared<ov::op::v1::Select>(cb.mask, select_true, select_false);
    causal_float->set_friendly_name("model.causal_mask");

    auto unsqueeze_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});

    auto causal_4d = std::make_shared<ov::opset11::Unsqueeze>(causal_float, unsqueeze_axes);
    causal_4d->set_friendly_name("model.causal_mask_4d");

    auto combined = std::make_shared<ov::opset11::Add>(padding_4d, causal_4d);
    combined->set_friendly_name("model.mask_4d");

    return combined->output(0);
}

CachePositionResult make_cache_position_ids(const ov::Output<ov::Node>& input_ids,
                                            const ov::Output<ov::Node>& kv_cache_beam_gather,
                                            const std::string& prefix) {
    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(input_ids, ov::element::i64);
    ids_shape->set_friendly_name(prefix + "ids_shape");
    auto seq_len =
        std::make_shared<ov::opset11::Gather>(ids_shape,
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1}),
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    seq_len->set_friendly_name(prefix + "seq_len");

    // kv_seq_len = ShapeOf(beam_gather)[2] — root of the CachePositionInput pattern
    auto kv_shape = std::make_shared<ov::opset11::ShapeOf>(kv_cache_beam_gather, ov::element::i64);
    kv_shape->set_friendly_name(prefix + "kv_shape");
    auto kv_seq_len =
        std::make_shared<ov::opset11::Gather>(kv_shape,
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {2}),
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    kv_seq_len->set_friendly_name(prefix + "kv_seq_len");

    // CachePositionInput pattern: Gather -> Add -> Range -> Unsqueeze -> Tile
    auto total_seq_len = std::make_shared<ov::opset11::Add>(kv_seq_len, seq_len);
    total_seq_len->set_friendly_name(prefix + "total_seq_len");

    auto cache_positions = std::make_shared<ov::op::v4::Range>(
        kv_seq_len->output(0),
        total_seq_len->output(0),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1})->output(0),
        ov::element::i64);
    cache_positions->set_friendly_name(prefix + "cache_positions");

    auto cache_pos_unsq =
        std::make_shared<ov::opset11::Unsqueeze>(cache_positions,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    cache_pos_unsq->set_friendly_name(prefix + "cache_pos_unsq");

    auto batch_dim_for_tile =
        std::make_shared<ov::opset11::Gather>(ids_shape,
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    batch_dim_for_tile->set_friendly_name(prefix + "batch_for_tile");

    auto tile_repeats = std::make_shared<ov::opset11::Concat>(
        ov::OutputVector{batch_dim_for_tile->output(0),
                         ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1})->output(0)},
        0);
    tile_repeats->set_friendly_name(prefix + "tile_repeats");

    auto position_ids = std::make_shared<ov::op::v0::Tile>(cache_pos_unsq, tile_repeats);
    position_ids->set_friendly_name(prefix + "position_ids");

    return {position_ids->output(0),
            total_seq_len->output(0),
            seq_len->output(0),
            cache_pos_unsq->output(0),
            ids_shape->output(0)};
}

ov::Output<ov::Node> make_whisper_causal_mask(const CachePositionResult& cache_pos,
                                              const std::string& prefix,
                                              bool boolean_output) {
    // kv_idx: Range -> 3x Unsqueeze -> [1, 1, 1, total_seq]
    auto mask_range = std::make_shared<ov::op::v4::Range>(
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0})->output(0),
        cache_pos.total_seq_len,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1})->output(0),
        ov::element::i64);
    mask_range->set_friendly_name(prefix + "mask_range");

    auto kv_unsq1 =
        std::make_shared<ov::opset11::Unsqueeze>(mask_range,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    kv_unsq1->set_friendly_name(prefix + "kv_unsq1");
    auto kv_unsq2 =
        std::make_shared<ov::opset11::Unsqueeze>(kv_unsq1,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    kv_unsq2->set_friendly_name(prefix + "kv_unsq2");
    auto kv_unsq3 =
        std::make_shared<ov::opset11::Unsqueeze>(kv_unsq2,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {2}));
    kv_unsq3->set_friendly_name(prefix + "kv_unsq3");

    // q_idx: cache_pos_unsq -> 2x Unsqueeze -> [1, 1, seq, 1]
    auto q_unsq1 =
        std::make_shared<ov::opset11::Unsqueeze>(cache_pos.cache_pos_unsq,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    q_unsq1->set_friendly_name(prefix + "q_unsq1");
    auto q_unsq2 =
        std::make_shared<ov::opset11::Unsqueeze>(q_unsq1,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {3}));
    q_unsq2->set_friendly_name(prefix + "q_unsq2");

    auto causal_bool = std::make_shared<ov::op::v1::LessEqual>(kv_unsq3, q_unsq2);
    causal_bool->set_friendly_name(prefix + "causal_mask_bool");

    auto batch_dim_b =
        std::make_shared<ov::opset11::Gather>(cache_pos.ids_shape,
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto seq_len_1d =
        std::make_shared<ov::opset11::Reshape>(cache_pos.seq_len,
                                               ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                               false);
    auto total_seq_1d =
        std::make_shared<ov::opset11::Unsqueeze>(cache_pos.total_seq_len,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto broadcast_shape = std::make_shared<ov::opset11::Concat>(
        ov::OutputVector{batch_dim_b->output(0),
                         ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1})->output(0),
                         seq_len_1d->output(0),
                         total_seq_1d->output(0)},
        0);
    auto causal_broadcast =
        std::make_shared<ov::op::v3::Broadcast>(causal_bool, broadcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
    causal_broadcast->set_friendly_name(prefix + "causal_mask_broadcast");

    // Float path: Select to f32. Boolean path: skip the Select and feed the bool
    // tensor to the Slice, exercising NPUW's boolean mask handling for Whisper.
    ov::Output<ov::Node> mask_for_slice;
    if (boolean_output) {
        mask_for_slice = causal_broadcast->output(0);
    } else {
        // Always f32 — NPUW's AttentionMask matchers inject f32 nodes
        auto select_true = ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
        auto select_false =
            ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {kAttentionMaskPaddingFP16Min});
        auto causal_float = std::make_shared<ov::op::v1::Select>(causal_broadcast, select_true, select_false);
        causal_float->set_friendly_name(prefix + "causal_mask");
        mask_for_slice = causal_float->output(0);
    }

    // Structural no-op Slice — AttentionMaskInput (prefill) needs Slice -> SDPA input[3]
    auto slice_start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto slice_stop =
        std::make_shared<ov::opset11::Reshape>(cache_pos.total_seq_len,
                                               ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                               false);
    auto slice_step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto slice_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {3});
    auto causal_sliced =
        std::make_shared<ov::opset11::Slice>(mask_for_slice, slice_start, slice_stop, slice_step, slice_axes);
    causal_sliced->set_friendly_name(prefix + "causal_mask_sliced");

    return causal_sliced->output(0);
}

ov::Output<ov::Node> make_causal_mask_boolean(const ov::Output<ov::Node>& input_ids_output,
                                              const ov::Output<ov::Node>& attention_mask_output,
                                              ov::element::Type /*unused*/) {
    auto cb = make_causal_bool(input_ids_output, attention_mask_output, "model.cmb.");

    auto unsq_2d = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto causal_4d = std::make_shared<ov::opset11::Unsqueeze>(cb.mask, unsq_2d);
    causal_4d->set_friendly_name("model.cmb.causal_4d");

    // Boolean padding mask: attention_mask -> bool [batch, 1, 1, total_seq].
    auto attn_bool = std::make_shared<ov::opset11::Convert>(attention_mask_output, ov::element::boolean);
    attn_bool->set_friendly_name("model.cmb.attn_bool");
    auto pad_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 2});
    auto attn_bool_4d = std::make_shared<ov::opset11::Unsqueeze>(attn_bool, pad_axes);
    attn_bool_4d->set_friendly_name("model.cmb.attn_bool_4d");

    auto combined = std::make_shared<ov::op::v13::BitwiseAnd>(causal_4d, attn_bool_4d);
    combined->set_friendly_name("model.cmb.combined");

    return combined->output(0);
}

ov::Output<ov::Node> make_sliding_window_mask(const ov::Output<ov::Node>& input_ids_output,
                                              const ov::Output<ov::Node>& attention_mask_output,
                                              ov::element::Type prec,
                                              size_t window_size) {
    auto padding_4d = make_padding_mask(attention_mask_output, prec);
    auto cb = make_causal_bool(input_ids_output, attention_mask_output, "model.sw.");

    // Sliding: kv > q - window_size (within window)
    auto window_const =
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(window_size)});
    auto lower_bound = std::make_shared<ov::opset11::Subtract>(cb.q_col, window_const);
    lower_bound->set_friendly_name("model.sw.lower_bound");

    auto sliding_bool = std::make_shared<ov::op::v1::Greater>(cb.kv_row, lower_bound);
    sliding_bool->set_friendly_name("model.sw.sliding_bool");

    // Combined: attend iff causal AND sliding
    auto combined_bool = std::make_shared<ov::op::v1::LogicalAnd>(cb.mask, sliding_bool);
    combined_bool->set_friendly_name("model.sw.combined_bool");

    auto select_true = ov::opset11::Constant::create(prec, ov::Shape{}, {0.0f});
    auto select_false = ov::opset11::Constant::create(prec, ov::Shape{}, {kAttentionMaskPadding});

    auto mask_float = std::make_shared<ov::op::v1::Select>(combined_bool, select_true, select_false);
    mask_float->set_friendly_name("model.sw.mask");

    auto unsqueeze_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto mask_4d = std::make_shared<ov::opset11::Unsqueeze>(mask_float, unsqueeze_axes);
    mask_4d->set_friendly_name("model.sw.mask_4d");

    auto combined = std::make_shared<ov::opset11::Add>(padding_4d, mask_4d);
    combined->set_friendly_name("model.sw.combined_mask_4d");

    return combined->output(0);
}

namespace {

std::shared_ptr<ov::Node> unsq1(const ov::Output<ov::Node>& in, int64_t axis) {
    auto ax = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{axis});
    return std::make_shared<ov::opset11::Unsqueeze>(in, ax);
}

/// Shared scaffolding for the boolean SWA builders (Phi-3 and Gemma-4 shapes):
/// scalar seq_len, scalar past_kv_len routed through Gather + Squeeze (the
/// matcher anchor), the K-side row [1, 1, 1, total_seq] (Range -> 3x Unsqueeze)
/// and the boolean user-attention padding [batch, 1, 1, total_seq]. The two
/// matchers differ only in the Q-side construction.
struct SwaParts {
    std::shared_ptr<ov::Node> seq_len;       ///< i64 scalar
    std::shared_ptr<ov::Node> past_kv_len;   ///< i64 scalar, Squeeze(Gather(...))
    std::shared_ptr<ov::Node> step;          ///< i64 scalar = 1
    std::shared_ptr<ov::Node> zero;          ///< i64 scalar = 0
    std::shared_ptr<ov::Node> k_row;         ///< i64 [1, 1, 1, total_seq]
    std::shared_ptr<ov::Node> attn_bool_4d;  ///< bool [batch, 1, 1, total_seq]
};

SwaParts make_swa_parts(const ov::Output<ov::Node>& seq_source,
                        const ov::Output<ov::Node>& attention_mask,
                        const std::string& p) {
    SwaParts r;

    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(seq_source, ov::element::i64);
    ids_shape->set_friendly_name(p + "ids_shape");
    auto mask_shape = std::make_shared<ov::opset11::ShapeOf>(attention_mask, ov::element::i64);
    mask_shape->set_friendly_name(p + "mask_shape");

    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto seq_len = std::make_shared<ov::opset11::Gather>(ids_shape, idx1, axis0);
    seq_len->set_friendly_name(p + "seq_len");
    auto total_seq = std::make_shared<ov::opset11::Gather>(mask_shape, idx1, axis0);
    total_seq->set_friendly_name(p + "total_seq");

    // past_kv_len = total_seq - seq_len, then routed through a Gather so the
    // matcher anchor (Gather → optional Squeeze) is satisfied.
    auto past_kv_diff = std::make_shared<ov::opset11::Subtract>(total_seq, seq_len);
    past_kv_diff->set_friendly_name(p + "past_kv_diff");
    auto rank1_shape = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto past_kv_1d = std::make_shared<ov::opset11::Reshape>(past_kv_diff, rank1_shape, false);
    past_kv_1d->set_friendly_name(p + "past_kv_1d");
    auto past_kv_len_gather = std::make_shared<ov::opset11::Gather>(past_kv_1d, axis0, axis0);
    past_kv_len_gather->set_friendly_name(p + "past_kv_len");
    // Optional Squeeze in matcher — present here (no axes form).
    auto past_kv_len = std::make_shared<ov::opset11::Squeeze>(past_kv_len_gather);
    past_kv_len->set_friendly_name(p + "past_kv_len_sq");

    r.step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    r.zero = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});

    // K-side full context length, a separate Add node from the Q-side one,
    // with past_kv_len as the first operand. Both details match the real
    // exports and both matter: the matcher pairs its past_kv_len anchor with
    // operand 0 first when permuting the commutative Add, and seq_len is also
    // a bare Gather, so putting it first mis-binds the anchor.
    auto full_ctx_k = std::make_shared<ov::opset11::Add>(past_kv_len, seq_len);
    full_ctx_k->set_friendly_name(p + "full_ctx_k");
    auto k_range = std::make_shared<ov::op::v4::Range>(r.zero, full_ctx_k, r.step, ov::element::i64);
    k_range->set_friendly_name(p + "k_range");

    // 3x Unsqueeze chain — the matchers' unsqueeze_sequence(). K goes to row
    // [1, 1, 1, total_seq]. No Convert (matches the optional<Convert> absent).
    auto k_row = unsq1(unsq1(unsq1(k_range, 0), 0), 0);
    k_row->set_friendly_name(p + "k_row");

    // Boolean padding (any-bool input to first BitwiseAnd) — keep things in bool domain.
    auto attn_bool = std::make_shared<ov::opset11::Convert>(attention_mask, ov::element::boolean);
    attn_bool->set_friendly_name(p + "attn_bool");
    auto pad_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 2});
    auto attn_bool_4d = std::make_shared<ov::opset11::Unsqueeze>(attn_bool, pad_axes);
    attn_bool_4d->set_friendly_name(p + "attn_bool_4d");

    r.seq_len = seq_len;
    r.past_kv_len = past_kv_len;
    r.k_row = k_row;
    r.attn_bool_4d = attn_bool_4d;
    return r;
}

/// Shared SWA epilogue: lower bound = Add(q_col, negative window constant),
/// then BitwiseAnd(user_attention, sliding) → BitwiseAnd(_, causal). Both
/// matchers anchor on this exact op sequence (incl. operand order).
ov::Output<ov::Node> combine_sliding_and_causal(const SwaParts& parts,
                                                const std::shared_ptr<ov::Node>& q_col,
                                                const std::shared_ptr<ov::Node>& neg_window,
                                                const std::string& p) {
    auto query_left_bound = std::make_shared<ov::opset11::Add>(q_col, neg_window);
    query_left_bound->set_friendly_name(p + "query_left_bound");

    auto sliding_bool = std::make_shared<ov::op::v1::Greater>(parts.k_row, query_left_bound);
    sliding_bool->set_friendly_name(p + "sliding_bool");

    // Multiple BitwiseAnd: BitwiseAnd(user_attention, sliding) → BitwiseAnd(_, causal).
    // First operand is the user/padding attention mask (matcher's any_input slot).
    auto sliding_and_user_attention = std::make_shared<ov::op::v13::BitwiseAnd>(parts.attn_bool_4d, sliding_bool);
    sliding_and_user_attention->set_friendly_name(p + "sliding_and_user_attention");

    auto causal_bool = std::make_shared<ov::op::v1::LessEqual>(parts.k_row, q_col);
    causal_bool->set_friendly_name(p + "causal_bool");

    auto sliding_and_causal = std::make_shared<ov::op::v13::BitwiseAnd>(sliding_and_user_attention, causal_bool);
    sliding_and_causal->set_friendly_name(p + "sliding_and_causal");

    return sliding_and_causal->output(0);
}

}  // namespace

ov::Output<ov::Node> make_sliding_window_mask_phi3(const ov::Output<ov::Node>& input_ids_output,
                                                   const ov::Output<ov::Node>& attention_mask_output,
                                                   ov::element::Type /*unused*/,
                                                   size_t window_size) {
    const std::string p = "model.swp.";
    auto parts = make_swa_parts(input_ids_output, attention_mask_output, p);

    // Phi-3 / Gemma-2 / Gemma-3 Q side: Range(past_kv_len, past_kv_len + seq_len)
    // — absolute positions, Range STARTS at past_kv_len.
    auto full_ctx_q = std::make_shared<ov::opset11::Add>(parts.past_kv_len, parts.seq_len);
    full_ctx_q->set_friendly_name(p + "full_ctx_q");
    auto q_range = std::make_shared<ov::op::v4::Range>(parts.past_kv_len, full_ctx_q, parts.step, ov::element::i64);
    q_range->set_friendly_name(p + "q_range");
    // Q column [1, 1, seq, 1] via the 3x Unsqueeze the matcher expects.
    auto q_col = unsq1(unsq1(unsq1(q_range, 1), 0), 0);
    q_col->set_friendly_name(p + "q_col");

    // Lower bound = Add(q_col, neg_window) with NEGATIVE constant (not Subtract).
    auto neg_window =
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {-static_cast<int64_t>(window_size)});
    neg_window->set_friendly_name(p + "neg_window");

    return combine_sliding_and_causal(parts, q_col, neg_window, p);
}

ov::Output<ov::Node> make_sliding_window_mask_gemma4(const ov::Output<ov::Node>& seq_source,
                                                     const ov::Output<ov::Node>& attention_mask_output,
                                                     ov::element::Type /*unused*/,
                                                     size_t window_size) {
    const std::string p = "model.swg.";
    auto parts = make_swa_parts(seq_source, attention_mask_output, p);

    // Gemma-4 Q side: cache_position = Range(0, seq_len) + past_kv_len —
    // the explicit Add AFTER the Range is what distinguishes this pattern
    // from Phi-3's Range(past_kv_len, ...) and what Gemma4SlidingMaskMatcher
    // anchors on (operand order: range first, past_kv_len second).
    auto q_range = std::make_shared<ov::op::v4::Range>(parts.zero, parts.seq_len, parts.step, ov::element::i64);
    q_range->set_friendly_name(p + "q_range");
    auto cache_position = std::make_shared<ov::opset11::Add>(q_range, parts.past_kv_len);
    cache_position->set_friendly_name(p + "cache_position");
    auto q_col = unsq1(unsq1(unsq1(cache_position, 1), 0), 0);
    q_col->set_friendly_name(p + "q_col");

    // Real Gemma-4 exports the window as a [1,1,1,1] negative constant.
    auto neg_window = ov::opset11::Constant::create(ov::element::i64,
                                                    ov::Shape{1, 1, 1, 1},
                                                    {-static_cast<int64_t>(window_size)});
    neg_window->set_friendly_name(p + "neg_window");

    return combine_sliding_and_causal(parts, q_col, neg_window, p);
}

ov::Output<ov::Node> make_sliding_window_mask_phi3_legacy(const ov::Output<ov::Node>& seq_source,
                                                          const ov::Output<ov::Node>& attention_mask_output,
                                                          ov::element::Type prec,
                                                          size_t window_size) {
    const std::string p = "model.swl.";

    // K side: Range(0, Gather(ShapeOf(attention_mask), 1)) → Convert → Convert.
    // OldPhi3SlidingMaskMatcher requires the ShapeOf fed DIRECTLY by the
    // attention_mask Parameter and the double Convert on the key range.
    auto mask_shape = std::make_shared<ov::opset11::ShapeOf>(attention_mask_output, ov::element::i64);
    mask_shape->set_friendly_name(p + "mask_shape");
    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto atten_len = std::make_shared<ov::opset11::Gather>(mask_shape, idx1, axis0);
    atten_len->set_friendly_name(p + "atten_len");

    auto zero = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto k_range = std::make_shared<ov::op::v4::Range>(zero, atten_len, step, ov::element::i32);
    k_range->set_friendly_name(p + "k_range");
    auto k_range_i64 = std::make_shared<ov::opset11::Convert>(k_range, ov::element::i64);
    k_range_i64->set_friendly_name(p + "k_range_i64");
    auto k_range_f32 = std::make_shared<ov::opset11::Convert>(k_range_i64, ov::element::f32);
    k_range_f32->set_friendly_name(p + "k_range_f32");

    // Q side: the matcher anchors seq length on ShapeOf(Parameter) → Gather
    // (it binds "position_ids" to ANY Parameter — seq_source qualifies) and
    // past_kv_len on a bare Gather (no Squeeze in this pattern).
    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(seq_source, ov::element::i64);
    ids_shape->set_friendly_name(p + "ids_shape");
    auto pos_len = std::make_shared<ov::opset11::Gather>(ids_shape, idx1, axis0);
    pos_len->set_friendly_name(p + "pos_len");

    auto past_kv_diff = std::make_shared<ov::opset11::Subtract>(atten_len, pos_len);
    past_kv_diff->set_friendly_name(p + "past_kv_diff");
    auto rank1_shape = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto past_kv_1d = std::make_shared<ov::opset11::Reshape>(past_kv_diff, rank1_shape, false);
    past_kv_1d->set_friendly_name(p + "past_kv_1d");
    auto past_kv_len = std::make_shared<ov::opset11::Gather>(past_kv_1d, axis0, axis0);
    past_kv_len->set_friendly_name(p + "past_kv_len");

    auto full_ctx = std::make_shared<ov::opset11::Add>(past_kv_len, pos_len);
    full_ctx->set_friendly_name(p + "full_ctx");
    // f32 Q range (4.51 export computed the bounds in float domain).
    auto q_range = std::make_shared<ov::op::v4::Range>(past_kv_len, full_ctx, step, ov::element::f32);
    q_range->set_friendly_name(p + "q_range");
    // Column via Reshape[-1, 1] — NOT Unsqueeze — in this pattern.
    auto col_shape = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 1});
    auto q_col = std::make_shared<ov::opset11::Reshape>(q_range, col_shape, false);
    q_col->set_friendly_name(p + "q_col");

    auto neg_window =
        ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {-static_cast<float>(window_size)});
    neg_window->set_friendly_name(p + "neg_window");
    auto query_left_bound = std::make_shared<ov::opset11::Add>(q_col, neg_window);
    query_left_bound->set_friendly_name(p + "query_left_bound");

    // INVERTED domain: true = token must NOT be attended.
    auto forget_left = std::make_shared<ov::op::v1::LessEqual>(k_range_f32, query_left_bound);
    forget_left->set_friendly_name(p + "forget_left");
    auto look_future = std::make_shared<ov::op::v1::Greater>(k_range_f32, q_col);
    look_future->set_friendly_name(p + "look_future");
    auto inv_sliding = std::make_shared<ov::op::v13::BitwiseOr>(look_future, forget_left);
    inv_sliding->set_friendly_name(p + "inv_sliding");

    // 4.51 consumers select the padding value where the inverted mask is true.
    auto masked = ov::opset11::Constant::create(prec, ov::Shape{}, {kAttentionMaskPadding});
    auto attend = ov::opset11::Constant::create(prec, ov::Shape{}, {0.0f});
    auto mask_float = std::make_shared<ov::op::v1::Select>(inv_sliding, masked, attend);
    mask_float->set_friendly_name(p + "mask");

    return mask_float->output(0);
}

ov::Output<ov::Node> make_vlm_bidirectional_modifier(const ov::Output<ov::Node>& base_mask,
                                                     const ov::Output<ov::Node>& token_type_ids_output,
                                                     const ov::Output<ov::Node>& seq_source,
                                                     ov::element::Type prec) {
    auto image_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});

    // K axis: full token_type_ids [batch, total_seq] → is_image → [batch, 1, 1, total_seq]
    auto k_is_image = std::make_shared<ov::op::v1::Equal>(token_type_ids_output, image_const);
    k_is_image->set_friendly_name("model.tti.k_is_image");
    auto k_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2});
    auto k_is_image_4d = std::make_shared<ov::opset11::Unsqueeze>(k_is_image, k_axes);
    k_is_image_4d->set_friendly_name("model.tti.k_is_image_4d");

    // Q axis: slice last seq_len entries from token_type_ids using offset = total_seq - seq_len.
    auto tti_shape = std::make_shared<ov::opset11::ShapeOf>(token_type_ids_output, ov::element::i64);
    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(seq_source, ov::element::i64);
    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto total_seq = std::make_shared<ov::opset11::Gather>(tti_shape, idx1, axis0);
    auto seq_len = std::make_shared<ov::opset11::Gather>(ids_shape, idx1, axis0);
    auto offset = std::make_shared<ov::opset11::Subtract>(total_seq, seq_len);
    offset->set_friendly_name("model.tti.offset");

    auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto seq_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto offset_1d = std::make_shared<ov::opset11::Unsqueeze>(offset, axis0);
    auto total_seq_1d = std::make_shared<ov::opset11::Unsqueeze>(total_seq, axis0);

    auto q_tti_slice =
        std::make_shared<ov::op::v8::Slice>(token_type_ids_output, offset_1d, total_seq_1d, step, seq_axis);
    q_tti_slice->set_friendly_name("model.tti.q_slice");

    auto q_is_image = std::make_shared<ov::op::v1::Equal>(q_tti_slice, image_const);
    q_is_image->set_friendly_name("model.tti.q_is_image");
    auto q_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {1, 3});
    auto q_is_image_4d = std::make_shared<ov::opset11::Unsqueeze>(q_is_image, q_axes);
    q_is_image_4d->set_friendly_name("model.tti.q_is_image_4d");

    // Both Q and K are image tokens -> bidirectional attention
    auto both_image = std::make_shared<ov::op::v1::LogicalAnd>(q_is_image_4d, k_is_image_4d);
    both_image->set_friendly_name("model.tti.both_image");

    // Select needs matching then/else types: boolean base masks (e.g. the Phi-3
    // style SWA pattern) stay in the bool domain with attend = true.
    const bool bool_base = base_mask.get_element_type() == ov::element::boolean;
    auto attend = bool_base ? ov::opset11::Constant::create(ov::element::boolean, ov::Shape{}, {true})
                            : ov::opset11::Constant::create(prec, ov::Shape{}, {0.0f});
    auto result = std::make_shared<ov::op::v1::Select>(both_image, attend, base_mask);
    result->set_friendly_name("model.tti.modified_mask");

    return result->output(0);
}

}  // namespace npuw
}  // namespace test
}  // namespace ov

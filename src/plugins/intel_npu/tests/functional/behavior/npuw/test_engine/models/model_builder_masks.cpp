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

    // Float path: Select to f32. Boolean path: skip Select, feed bool tensor to Slice
    // so the boolean handler at prepare_whisper_model.cpp:140 is exercised.
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

ov::Output<ov::Node> make_sliding_window_mask_phi3(const ov::Output<ov::Node>& input_ids_output,
                                                   const ov::Output<ov::Node>& attention_mask_output,
                                                   ov::element::Type /*unused*/,
                                                   size_t window_size) {
    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(input_ids_output, ov::element::i64);
    ids_shape->set_friendly_name("model.swp.ids_shape");
    auto mask_shape = std::make_shared<ov::opset11::ShapeOf>(attention_mask_output, ov::element::i64);
    mask_shape->set_friendly_name("model.swp.mask_shape");

    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto seq_len = std::make_shared<ov::opset11::Gather>(ids_shape, idx1, axis0);
    seq_len->set_friendly_name("model.swp.seq_len");
    auto total_seq = std::make_shared<ov::opset11::Gather>(mask_shape, idx1, axis0);
    total_seq->set_friendly_name("model.swp.total_seq");

    // past_kv_len = total_seq - seq_len, then routed through a Gather so the
    // Phi3 matcher anchor (Gather → optional Squeeze) is satisfied.
    auto past_kv_diff = std::make_shared<ov::opset11::Subtract>(total_seq, seq_len);
    past_kv_diff->set_friendly_name("model.swp.past_kv_diff");
    auto rank1_shape = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto past_kv_1d = std::make_shared<ov::opset11::Reshape>(past_kv_diff, rank1_shape, false);
    past_kv_1d->set_friendly_name("model.swp.past_kv_1d");
    auto past_kv_len_gather = std::make_shared<ov::opset11::Gather>(past_kv_1d, axis0, axis0);
    past_kv_len_gather->set_friendly_name("model.swp.past_kv_len");
    // Optional Squeeze in matcher — present here (no axes form).
    auto past_kv_len = std::make_shared<ov::opset11::Squeeze>(past_kv_len_gather);
    past_kv_len->set_friendly_name("model.swp.past_kv_len_sq");

    auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto zero = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});

    // Two separate Add nodes for total_ctx_len — Phi3 matcher pattern requires this.
    auto full_ctx_q = std::make_shared<ov::opset11::Add>(past_kv_len, seq_len);
    full_ctx_q->set_friendly_name("model.swp.full_ctx_q");
    auto full_ctx_k = std::make_shared<ov::opset11::Add>(seq_len, past_kv_len);
    full_ctx_k->set_friendly_name("model.swp.full_ctx_k");

    auto q_range = std::make_shared<ov::op::v4::Range>(past_kv_len, full_ctx_q, step, ov::element::i64);
    q_range->set_friendly_name("model.swp.q_range");
    auto k_range = std::make_shared<ov::op::v4::Range>(zero, full_ctx_k, step, ov::element::i64);
    k_range->set_friendly_name("model.swp.k_range");

    // 3x Unsqueeze chain on each range — Phi3 matcher's unsqueeze_sequence().
    // Q goes to column [1, 1, seq, 1]; K goes to row [1, 1, 1, total_seq].
    auto unsq_axis = [&](int64_t v) {
        return ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{v});
    };
    auto q_u1 = std::make_shared<ov::opset11::Unsqueeze>(q_range, unsq_axis(1));
    auto q_u2 = std::make_shared<ov::opset11::Unsqueeze>(q_u1, unsq_axis(0));
    auto q_col = std::make_shared<ov::opset11::Unsqueeze>(q_u2, unsq_axis(0));
    q_col->set_friendly_name("model.swp.q_col");

    auto k_u1 = std::make_shared<ov::opset11::Unsqueeze>(k_range, unsq_axis(0));
    auto k_u2 = std::make_shared<ov::opset11::Unsqueeze>(k_u1, unsq_axis(0));
    auto k_row = std::make_shared<ov::opset11::Unsqueeze>(k_u2, unsq_axis(0));
    k_row->set_friendly_name("model.swp.k_row");

    // No Convert here — keep ranges as i64 (matches the optional<Convert> being absent).
    // Lower bound = Add(q_col, neg_window) with NEGATIVE constant (not Subtract).
    auto neg_window = ov::opset11::Constant::create(ov::element::i64,
                                                    ov::Shape{},
                                                    {-static_cast<int64_t>(window_size)});
    neg_window->set_friendly_name("model.swp.neg_window");
    auto query_left_bound = std::make_shared<ov::opset11::Add>(q_col, neg_window);
    query_left_bound->set_friendly_name("model.swp.query_left_bound");

    auto sliding_bool = std::make_shared<ov::op::v1::Greater>(k_row, query_left_bound);
    sliding_bool->set_friendly_name("model.swp.sliding_bool");

    // Boolean padding (any-bool input to first BitwiseAnd) — keep things in bool domain.
    auto attn_bool = std::make_shared<ov::opset11::Convert>(attention_mask_output, ov::element::boolean);
    attn_bool->set_friendly_name("model.swp.attn_bool");
    auto pad_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 2});
    auto attn_bool_4d = std::make_shared<ov::opset11::Unsqueeze>(attn_bool, pad_axes);
    attn_bool_4d->set_friendly_name("model.swp.attn_bool_4d");

    // Multiple BitwiseAnd: BitwiseAnd(any_bool, sliding) → BitwiseAnd(_, causal).
    auto sliding_and_true = std::make_shared<ov::op::v13::BitwiseAnd>(attn_bool_4d, sliding_bool);
    sliding_and_true->set_friendly_name("model.swp.sliding_and_true");

    auto causal_bool = std::make_shared<ov::op::v1::LessEqual>(k_row, q_col);
    causal_bool->set_friendly_name("model.swp.causal_bool");

    auto sliding_and_causal = std::make_shared<ov::op::v13::BitwiseAnd>(sliding_and_true, causal_bool);
    sliding_and_causal->set_friendly_name("model.swp.sliding_and_causal");

    return sliding_and_causal->output(0);
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

    auto attend = ov::opset11::Constant::create(prec, ov::Shape{}, {0.0f});
    auto result = std::make_shared<ov::op::v1::Select>(both_image, attend, base_mask);
    result->set_friendly_name("model.tti.modified_mask");

    return result->output(0);
}

}  // namespace npuw
}  // namespace test
}  // namespace ov

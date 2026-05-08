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

ov::Output<ov::Node> make_causal_mask(const ov::Output<ov::Node>& input_ids_output,
                                      const ov::Output<ov::Node>& attention_mask_output,
                                      ov::element::Type prec) {
    auto padding_4d = make_padding_mask(attention_mask_output, prec);

    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(input_ids_output, ov::element::i64);
    ids_shape->set_friendly_name("model.ids_shape");

    auto mask_shape_node = std::make_shared<ov::opset11::ShapeOf>(attention_mask_output, ov::element::i64);
    mask_shape_node->set_friendly_name("model.mask_shape");

    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});

    auto seq_len_s = std::make_shared<ov::opset11::Gather>(ids_shape, idx1, gather_axis);
    seq_len_s->set_friendly_name("model.seq_len");

    auto total_seq_s = std::make_shared<ov::opset11::Gather>(mask_shape_node, idx1, gather_axis);
    total_seq_s->set_friendly_name("model.total_seq");

    auto offset = std::make_shared<ov::opset11::Subtract>(total_seq_s, seq_len_s);
    offset->set_friendly_name("model.causal_offset");

    auto range_start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto range_step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});

    auto kv_range = std::make_shared<ov::op::v4::Range>(range_start, total_seq_s, range_step, ov::element::i64);
    kv_range->set_friendly_name("model.kv_range");

    auto q_range = std::make_shared<ov::op::v4::Range>(range_start, seq_len_s, range_step, ov::element::i64);
    q_range->set_friendly_name("model.q_range");

    auto q_abs = std::make_shared<ov::opset11::Add>(q_range, offset);
    q_abs->set_friendly_name("model.q_abs_positions");

    auto axis_last = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto axis_first = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});

    auto q_col = std::make_shared<ov::opset11::Unsqueeze>(q_abs, axis_last);
    q_col->set_friendly_name("model.q_col");

    auto kv_row = std::make_shared<ov::opset11::Unsqueeze>(kv_range, axis_first);
    kv_row->set_friendly_name("model.kv_row");

    auto causal_bool = std::make_shared<ov::op::v1::LessEqual>(kv_row, q_col);
    causal_bool->set_friendly_name("model.causal_bool");

    auto select_true = ov::opset11::Constant::create(prec, ov::Shape{}, {0.0f});
    auto select_false = ov::opset11::Constant::create(prec, ov::Shape{}, {kAttentionMaskPadding});

    auto causal_float = std::make_shared<ov::op::v1::Select>(causal_bool, select_true, select_false);
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

ov::Output<ov::Node> make_whisper_causal_mask(const CachePositionResult& cache_pos, const std::string& prefix) {
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

    // Always f32 — NPUW's AttentionMask matchers inject f32 nodes
    auto select_true = ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto select_false = ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {kAttentionMaskPaddingFP16Min});
    auto causal_float = std::make_shared<ov::op::v1::Select>(causal_broadcast, select_true, select_false);
    causal_float->set_friendly_name(prefix + "causal_mask");

    // Structural no-op Slice — AttentionMaskInput (prefill) needs Slice -> SDPA input[3]
    auto slice_start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto slice_stop =
        std::make_shared<ov::opset11::Reshape>(cache_pos.total_seq_len,
                                               ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                               false);
    auto slice_step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto slice_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {3});
    auto causal_sliced =
        std::make_shared<ov::opset11::Slice>(causal_float, slice_start, slice_stop, slice_step, slice_axes);
    causal_sliced->set_friendly_name(prefix + "causal_mask_sliced");

    return causal_sliced->output(0);
}

}  // namespace npuw
}  // namespace test
}  // namespace ov

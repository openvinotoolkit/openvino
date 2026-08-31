// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_sliding_window_manager.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "infer_request_utils.hpp"
#include "llm_compiled_model_utils.hpp"
#include "logging.hpp"
#include "npuw_transformations/detect_causal_mask.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/itensor.hpp"
#include "util.hpp"

namespace {

struct MaskView {
    uint32_t row_dim = 0;
    uint32_t col_dim = 0;
    uint32_t past_width = 0;
    uint32_t row_pad = 0;
    float* data = nullptr;
};

MaskView get_mask_view(const ov::SoPtr<ov::ITensor>& mask_tensor,
                       uint32_t num_real_new_tokens,
                       const char* caller_name) {
    OPENVINO_ASSERT(mask_tensor->get_element_type() == ov::element::f32,
                    caller_name,
                    ": Attention mask tensor is expected to be f32, got: ",
                    mask_tensor->get_element_type());
    const auto& shape = mask_tensor->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, caller_name, ": Attention mask tensor rank must be >= 2, got shape: ", shape);

    const uint32_t row_dim = static_cast<uint32_t>(shape[shape.size() - 2]);
    const uint32_t col_dim = static_cast<uint32_t>(shape[shape.size() - 1]);
    OPENVINO_ASSERT(col_dim >= row_dim,
                    caller_name,
                    ": attention mask key axis (",
                    col_dim,
                    ") must be >= query axis (",
                    row_dim,
                    ")");
    OPENVINO_ASSERT(num_real_new_tokens <= row_dim,
                    caller_name,
                    ": num_real_new_tokens (",
                    num_real_new_tokens,
                    ") exceeds query axis (",
                    row_dim,
                    ")");

    MaskView mv;
    mv.row_dim = row_dim;
    mv.col_dim = col_dim;
    mv.past_width = col_dim - row_dim;
    mv.row_pad = row_dim - num_real_new_tokens;
    mv.data = mask_tensor->data<float>();
    return mv;
}

}  // namespace

ov::npuw::util::SwaLayout ov::npuw::util::detect_swa_layout(const std::shared_ptr<ov::Model>& model) {
    // Read back per-layer SDPA mask annotations (see NPUW_SDPA_MASK_RT_KEY).
    // Missing entries are treated as full/causal layers.
    std::vector<int64_t> layer_mask_annotations;
    std::vector<bool> layer_has_annotation;
    size_t num_annotated_layers = 0;

    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
        if (!sdpa) {
            continue;
        }
        size_t layer_idx = 0;
        if (!try_parse_self_attn_layer_idx(sdpa->get_friendly_name(), layer_idx)) {
            continue;
        }
        const auto& rt_info = sdpa->get_rt_info();
        const auto it = rt_info.find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
        if (it == rt_info.end()) {
            continue;
        }

        const int64_t encoded = it->second.as<int64_t>();
        if (layer_idx >= layer_mask_annotations.size()) {
            layer_mask_annotations.resize(layer_idx + 1, 0);
            layer_has_annotation.resize(layer_idx + 1, false);
        }

        if (layer_has_annotation[layer_idx]) {
            OPENVINO_ASSERT(layer_mask_annotations[layer_idx] == encoded,
                            "NPUW SWA: conflicting SDPA mask annotations for layer ",
                            layer_idx,
                            " (",
                            layer_mask_annotations[layer_idx],
                            " vs ",
                            encoded,
                            ").");
        } else {
            layer_mask_annotations[layer_idx] = encoded;
            layer_has_annotation[layer_idx] = true;
            ++num_annotated_layers;
        }
    }

    SwaLayout layout;

    if (num_annotated_layers == 0) {
        LOG_DEBUG("[SWA] No per-layer mask annotations found; Sliding Window Attention is disabled.");
        return layout;
    }

    // Layers absent from the annotation set are treated as full-attention.
    const size_t num_layers = layer_mask_annotations.size();

    std::vector<bool> layer_is_sliding(num_layers, false);
    int64_t detected_window = 0;
    bool has_detected_window = false;
    bool has_sliding = false;
    bool has_full = false;

    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        if (layer_has_annotation[layer_idx] && layer_mask_annotations[layer_idx] >= 0) {
            const int64_t encoded = layer_mask_annotations[layer_idx];
            layer_is_sliding[layer_idx] = true;
            has_sliding = true;
            if (!has_detected_window) {
                detected_window = encoded;
                has_detected_window = true;
            } else {
                OPENVINO_ASSERT(detected_window == encoded,
                                "NPUW SWA: inconsistent sliding-window sizes detected across layers (",
                                detected_window,
                                " vs ",
                                encoded,
                                "). Only a single, uniform window size is currently supported.");
            }
        } else {
            has_full = true;
        }
    }

    // Only enable the hybrid SWA pipeline for a genuine hybrid model: at least one sliding-window
    // layer AND at least one full/causal-attention layer. A model where every layer is uniformly
    // sliding (or uniformly full attention) doesn't need per-layer KV-cache window capping.
    if (!has_sliding || !has_full) {
        LOG_DEBUG("[SWA] Not a genuine hybrid model (has_sliding=" << has_sliding << ", has_full=" << has_full
                                                                   << "); Sliding Window Attention support disabled.");
        return layout;
    }

    OPENVINO_ASSERT(has_detected_window && detected_window > 0,
                    "NPUW SWA: invalid sliding window size detected: ",
                    detected_window);

    layout.window_size = static_cast<uint32_t>(detected_window);
    layout.layer_is_sliding = std::move(layer_is_sliding);

    std::string pattern;
    pattern.reserve(layout.layer_is_sliding.size());
    size_t num_sliding = 0;
    for (const bool is_sliding : layout.layer_is_sliding) {
        pattern.push_back(is_sliding ? 'S' : 'F');
        num_sliding += is_sliding ? 1 : 0;
    }
    LOG_INFO("[SWA] Sliding Window Attention is ENABLED: window_size="
             << layout.window_size << ", " << layout.layer_is_sliding.size() << " layers total, " << num_sliding
             << " sliding-window layer(s).");
    LOG_DEBUG("[SWA] Layer pattern (S=sliding, F=full attention): " << pattern);

    return layout;
}

void ov::npuw::util::write_swa_kv_slice_circular(ov::SoPtr<ov::ITensor> dst_tensor,
                                                 ov::SoPtr<ov::ITensor> src_new_kv,
                                                 uint32_t dst_kv_dim,
                                                 uint32_t src_kv_dim,
                                                 uint32_t num_stored_tokens_before,
                                                 uint32_t num_new_tokens) {
    const uint32_t capacity = static_cast<uint32_t>(dst_tensor->get_shape()[dst_kv_dim]);
    const uint32_t old_total = num_stored_tokens_before;
    const uint32_t new_total = old_total + num_new_tokens;
    const uint32_t new_valid = std::min(new_total, capacity);

    // Clamp by source length as well: source may already be capacity-limited.
    const uint32_t src_len = static_cast<uint32_t>(src_new_kv->get_shape()[src_kv_dim]);
    const uint32_t tokens_to_write = std::min({num_new_tokens, new_valid, src_len});

    if (tokens_to_write == 0) {
        return;
    }
    const uint32_t first_new_abs_pos = num_stored_tokens_before + (num_new_tokens - tokens_to_write);
    const uint32_t dst_start = first_new_abs_pos % capacity;

    auto src_slice = (src_len > tokens_to_write)
                         ? ov::npuw::util::make_tensor_slice(src_new_kv, src_kv_dim, src_len - tokens_to_write, src_len)
                         : src_new_kv;

    if (dst_start + tokens_to_write <= capacity) {
        auto dst_slice =
            ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, dst_start, dst_start + tokens_to_write);
        ov::npuw::util::copy_tensor_by_dim(src_slice, dst_slice, src_kv_dim, dst_kv_dim);
    } else {
        const uint32_t first_leg_len = capacity - dst_start;
        const uint32_t second_leg_len = tokens_to_write - first_leg_len;

        auto src_first_leg = ov::npuw::util::make_tensor_slice(src_slice, src_kv_dim, 0u, first_leg_len);
        auto dst_first_leg = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, dst_start, capacity);
        ov::npuw::util::copy_tensor_by_dim(src_first_leg, dst_first_leg, src_kv_dim, dst_kv_dim);

        auto src_second_leg = ov::npuw::util::make_tensor_slice(src_slice, src_kv_dim, first_leg_len, tokens_to_write);
        auto dst_second_leg = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, 0u, second_leg_len);
        ov::npuw::util::copy_tensor_by_dim(src_second_leg, dst_second_leg, src_kv_dim, dst_kv_dim);
    }
}

void ov::npuw::util::write_swa_kv_slice_left_aligned(ov::SoPtr<ov::ITensor> dst_tensor,
                                                     ov::SoPtr<ov::ITensor> src_new_kv,
                                                     uint32_t dst_kv_dim,
                                                     uint32_t src_kv_dim,
                                                     uint32_t num_stored_tokens_before,
                                                     uint32_t num_new_tokens) {
    // Left-aligned SWA policy keeps valid tokens packed at the beginning
    // After update, logical order to be preserved as:
    //   [surviving old tail | newest appended tokens], truncated to capacity.
    const uint32_t capacity = static_cast<uint32_t>(dst_tensor->get_shape()[dst_kv_dim]);
    const uint32_t old_total = num_stored_tokens_before;
    const uint32_t new_total = old_total + num_new_tokens;
    const uint32_t old_valid = std::min(old_total, capacity);
    const uint32_t new_valid = std::min(new_total, capacity);

    // Clamp against source length too. Some source tensors can hold fewer tokens
    // than num_new_tokens when they were capacity-limited earlier.
    const uint32_t src_len = static_cast<uint32_t>(src_new_kv->get_shape()[src_kv_dim]);
    const uint32_t tokens_to_write = std::min({num_new_tokens, new_valid, src_len});

    // keep: number of old tokens that remain visible after appending the new chunk.
    // If keep < old_valid, the window is saturated and we must shift surviving old
    // tokens to the front before appending new ones.
    const uint32_t keep = new_valid - tokens_to_write;
    const bool needs_shift = (keep > 0 && keep < old_valid);

    if (needs_shift && dst_kv_dim == 3u) {
        // Transposed-V (dim=3) partial-slice shifts degrade into many small remote
        // transfers. Use a full-buffer round-trip to keep transfer count low:
        //   1) copy full dst -> CPU tmp,
        //   2) rearrange entirely on CPU,
        //   3) copy full CPU tmp -> dst.
        // This avoids many fine-grained remote transactions in the shift phase.
        LOG_DEBUG("[SWA] Bulk-shifting KV buffer (dim=3): keeping last " << keep << " of " << old_valid
                                                                         << " old token(s), capacity=" << capacity);
        auto whole_tmp =
            ov::npuw::util::allocMem(dst_tensor->get_element_type(), dst_tensor->get_shape(), "CPU", nullptr);
        dst_tensor->copy_to(whole_tmp._ptr);  // single bulk contiguous transfer

        auto old_tail_cpu = ov::npuw::util::make_tensor_slice(whole_tmp, dst_kv_dim, old_valid - keep, old_valid);
        auto shift_tmp =
            ov::npuw::util::allocMem(dst_tensor->get_element_type(), old_tail_cpu->get_shape(), "CPU", nullptr);
        old_tail_cpu->copy_to(shift_tmp._ptr);  // isolate surviving tail before front overwrite
        auto dst_front_cpu = ov::npuw::util::make_tensor_slice(whole_tmp, dst_kv_dim, 0u, keep);
        ov::npuw::util::copy_tensor_by_dim(shift_tmp, dst_front_cpu, dst_kv_dim, dst_kv_dim);

        if (tokens_to_write > 0) {
            auto src_slice =
                (src_len > tokens_to_write)
                    ? ov::npuw::util::make_tensor_slice(src_new_kv, src_kv_dim, src_len - tokens_to_write, src_len)
                    : src_new_kv;
            auto dst_back_cpu = ov::npuw::util::make_tensor_slice(whole_tmp, dst_kv_dim, keep, keep + tokens_to_write);
            ov::npuw::util::copy_tensor_by_dim(src_slice, dst_back_cpu, src_kv_dim, dst_kv_dim);
        }

        whole_tmp->copy_to(dst_tensor._ptr);  // single bulk contiguous transfer back
        return;
    }

    if (needs_shift) {
        // Window saturated: move the surviving old tail to the front.
        // Use a temporary buffer to avoid overlapping in-place copy.
        LOG_DEBUG("[SWA] Shifting KV buffer: keeping last "
                  << keep << " of " << old_valid << " old token(s), dim=" << dst_kv_dim << ", capacity=" << capacity);
        auto old_tail = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, old_valid - keep, old_valid);
        auto tmp = ov::npuw::util::allocMem(dst_tensor->get_element_type(), old_tail->get_shape(), "CPU", nullptr);
        old_tail->copy_to(tmp._ptr);
        auto dst_front = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, 0u, keep);
        ov::npuw::util::copy_tensor_by_dim(tmp, dst_front, dst_kv_dim, dst_kv_dim);
    }

    if (tokens_to_write == 0) {
        return;
    }
    // Append newest source tail right after the preserved prefix.
    auto src_slice = (src_len > tokens_to_write)
                         ? ov::npuw::util::make_tensor_slice(src_new_kv, src_kv_dim, src_len - tokens_to_write, src_len)
                         : src_new_kv;
    auto dst_back = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, keep, keep + tokens_to_write);
    ov::npuw::util::copy_tensor_by_dim(src_slice, dst_back, src_kv_dim, dst_kv_dim);
}

void ov::npuw::util::fill_causal_sliding_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                              uint32_t num_stored_tokens_before,
                                              uint32_t num_real_new_tokens,
                                              uint32_t window_size) {
    const auto mask_view = get_mask_view(mask_tensor, num_real_new_tokens, "fill_causal_sliding_mask");
    OPENVINO_ASSERT(window_size > 0, "fill_causal_sliding_mask: window_size must be > 0");

    const uint32_t stored_tokens_before = num_stored_tokens_before;
    const uint32_t past_width = mask_view.past_width;
    const uint32_t row_dim = mask_view.row_dim;
    const uint32_t row_pad = mask_view.row_pad;
    const bool has_past_region = past_width > 0u;
    const bool is_past_saturated = has_past_region && stored_tokens_before >= past_width;
    const uint32_t wrap_slot = is_past_saturated ? (stored_tokens_before % past_width) : 0u;
    const int64_t stored_tokens_before_i64 = static_cast<int64_t>(stored_tokens_before);
    const int64_t past_width_i64 = static_cast<int64_t>(past_width);
    const int64_t window_i64 = static_cast<int64_t>(window_size);
    const int64_t row_pad_i64 = static_cast<int64_t>(row_pad);

    constexpr float kAttend = 0.0f;
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    // Layout per mask row:
    //   columns = [past circular slots][current-chunk columns]
    //           = [0 .. past_width-1] [past_width .. past_width+row_dim-1]
    //   If past_width == 0, this degenerates to current-chunk-only masking.
    //
    // Past circular mapping (slot -> absolute token index):
    //   1) Unsaturated (stored_tokens_before < past_width)
    //      valid prefix only:
    //      slot->abs: [0, 1, 2, ..., stored_tokens_before-1, invalid, ...]
    //   2) Saturated (stored_tokens_before >= past_width)
    //      wrap_slot = stored_tokens_before % past_width
    //      slot->abs is split into two contiguous ranges:
    //        [0, wrap_slot):          abs = (stored_tokens_before - wrap_slot) + slot
    //        [wrap_slot, past_width): abs = (stored_tokens_before - wrap_slot) + slot - past_width
    //      example: past_width=8, stored_tokens_before=11, wrap_slot=3
    //               slot->abs: [8,9,10,3,4,5,6,7]
    //
    // Visibility rule for a row at absolute position row_abs_pos:
    //   attend(abs) iff abs <= row_abs_pos AND row_abs_pos - abs < window_size
    //   => visible abs interval: [row_abs_pos - window_size + 1, row_abs_pos]
    //
    // Current-chunk area is a causal diagonal clipped by the same window:
    //   local key index local_c is visible in this row when
    //   local_c in [max(row_pad, row-window_size+1), row].
    auto fill_clamped_range = [](float* ptr,
                                 int64_t range_begin,
                                 int64_t range_end,
                                 int64_t domain_begin,
                                 int64_t domain_end,
                                 float fill_value) {
        const int64_t clamped_begin = std::max(range_begin, domain_begin);
        const int64_t clamped_end = std::min(range_end, domain_end);
        if (clamped_begin <= clamped_end) {
            std::fill_n(ptr + clamped_begin, static_cast<size_t>(clamped_end - clamped_begin + 1), fill_value);
        }
    };

    for (uint32_t row = 0; row < row_dim; ++row) {
        float* row_ptr = mask_view.data + static_cast<size_t>(row) * mask_view.col_dim;

        // Fill both regions with masked value first; then unmask only visible intervals.
        std::fill_n(row_ptr, past_width, kMasked);
        std::fill_n(row_ptr + past_width, row_dim, kMasked);

        const int64_t row_i64 = static_cast<int64_t>(row);
        const int64_t row_abs_pos = stored_tokens_before_i64 + (row_i64 - row_pad_i64);
        const int64_t min_visible_abs_pos = row_abs_pos - window_i64 + 1;
        const int64_t max_visible_abs_pos = row_abs_pos;

        // Past region [0, past_width).
        if (is_past_saturated) {
            // Saturated ring: at most two contiguous slot ranges can be visible,
            // one in [wrap_slot, past_width) and one in [0, wrap_slot).
            const int64_t ring_base_abs = stored_tokens_before_i64 - static_cast<int64_t>(wrap_slot);
            const int64_t older_segment_bias = ring_base_abs - past_width_i64;  // abs = older_segment_bias + slot

            // Segment 1: c in [wrap_slot, past_width-1].
            fill_clamped_range(row_ptr,
                               min_visible_abs_pos - older_segment_bias,
                               max_visible_abs_pos - older_segment_bias,
                               static_cast<int64_t>(wrap_slot),
                               past_width_i64 - 1,
                               kAttend);

            // Segment 2: c in [0, wrap_slot-1].
            if (wrap_slot > 0u) {
                fill_clamped_range(row_ptr,
                                   min_visible_abs_pos - ring_base_abs,
                                   max_visible_abs_pos - ring_base_abs,
                                   0,
                                   static_cast<int64_t>(wrap_slot) - 1,
                                   kAttend);
            }
        } else if (has_past_region && stored_tokens_before > 0u) {
            // Unsaturated prefix: slot index equals absolute position for valid slots.
            fill_clamped_range(row_ptr,
                               min_visible_abs_pos,
                               max_visible_abs_pos,
                               0,
                               static_cast<int64_t>(stored_tokens_before) - 1,
                               kAttend);
        }

        // Current chunk diagonal region [past_width, past_width + row_dim).
        // local_c must satisfy all constraints below:
        //   1) valid key in right-aligned chunk: local_c >= row_pad
        //   2) causal:                         local_c <= row
        //   3) window:                         row - local_c < window_size
        // => local_c in [max(row_pad, row-window+1), row].
        const int64_t local_begin = std::max(row_pad_i64, row_i64 - window_i64 + 1);
        const int64_t local_end = row_i64;
        fill_clamped_range(row_ptr + past_width, local_begin, local_end, 0, static_cast<int64_t>(row_dim) - 1, kAttend);
    }
}

void ov::npuw::util::fill_attention_masks(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                          const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                          uint32_t num_stored_tokens_before,
                                          uint32_t num_real_new_tokens,
                                          uint32_t window_size,
                                          const int64_t* token_type_ids_real) {
    const auto it = in_ports.find(ov::npuw::util::kSlidingWindowAttentionMaskParamName);
    if (it == in_ports.end()) {
        return;
    }
    auto mask_tensor = request->get_tensor(it->second);
    fill_causal_sliding_mask(mask_tensor, num_stored_tokens_before, num_real_new_tokens, window_size);
    if (token_type_ids_real != nullptr) {
        overlay_vision_bidirectional_mask(mask_tensor, token_type_ids_real, num_real_new_tokens);
    }
}

void ov::npuw::util::overlay_vision_bidirectional_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                                       const int64_t* token_type_ids_real,
                                                       uint32_t num_real_new_tokens) {
    if (num_real_new_tokens == 0) {
        return;
    }
    OPENVINO_ASSERT(token_type_ids_real != nullptr,
                    "overlay_vision_bidirectional_mask: token_type_ids_real must not be null");

    const auto mask_view = get_mask_view(mask_tensor, num_real_new_tokens, "overlay_vision_bidirectional_mask");

    constexpr float kAttend = 0.0f;
    constexpr int64_t kVisionTokenTypeId = 1;

    // This function makes vision-token runs bidirectional inside the current chunk.
    // We scan token_type_ids_real and find contiguous runs where token_type_id == 1.
    // For each run [run_start, run_end), we unmask a square block in the current-chunk
    // submatrix: rows run_start..run_end-1 and cols run_start..run_end-1.
    //
    // Example (current chunk only, V=vision, T=text):
    //   token_type_ids_real: [T, V, V, V, T, V, V]
    //   runs: [1,4), [5,7)
    //
    //   local cols ->    0 1 2 3 4 5 6
    //   local rows
    //              0(T): . . . . . . .
    //              1(V): . A A A . . .
    //              2(V): . A A A . . .
    //              3(V): . A A A . . .
    //              4(T): . . . . . . .
    //              5(V): . . . . . B B
    //              6(V): . . . . . B B
    //   A/B: cells forced to attend (0.0f) by this overlay.
    auto apply_vision_run = [&](uint32_t run_start, uint32_t run_end_exclusive) {
        const uint32_t run_length = run_end_exclusive - run_start;
        const uint32_t run_col_start = mask_view.past_width + mask_view.row_pad + run_start;
        for (uint32_t row_index = run_start; row_index < run_end_exclusive; ++row_index) {
            float* row_ptr = mask_view.data + static_cast<size_t>(mask_view.row_pad + row_index) * mask_view.col_dim;
            std::fill_n(row_ptr + run_col_start, run_length, kAttend);
        }
    };

    uint32_t token_index = 0;
    while (token_index < num_real_new_tokens) {
        if (token_type_ids_real[token_index] != kVisionTokenTypeId) {
            ++token_index;
            continue;
        }
        const uint32_t run_start = token_index;
        while (token_index < num_real_new_tokens && token_type_ids_real[token_index] == kVisionTokenTypeId) {
            ++token_index;
        }
        apply_vision_run(run_start, token_index);
    }
}

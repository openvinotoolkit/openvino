// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_sliding_window_manager.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

#include "infer_request_utils.hpp"
#include "llm_compiled_model_utils.hpp"
#include "logging.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/itensor.hpp"
#include "util.hpp"

namespace {

struct MaskView {
    uint32_t row_dim = 0;            // Query axis length
    uint32_t col_dim = 0;            // Key axis length
    uint32_t past_width = 0;         // Past-region column count: col_dim - row_dim
    uint32_t row_pad = 0;            // Unused leading rows in this call's chunk: row_dim - num_real_new_tokens
    ov::element::Type element_type;  // Actual mask tensor element type: f32 or f16
    void* data = nullptr;            // Base pointer to the mask tensor's data
};

MaskView get_mask_view(const ov::SoPtr<ov::ITensor>& mask_tensor,
                       uint32_t num_real_new_tokens,
                       const char* caller_name) {
    const auto element_type = mask_tensor->get_element_type();
    OPENVINO_ASSERT(element_type == ov::element::f32 || element_type == ov::element::f16,
                    caller_name,
                    ": Attention mask tensor is expected to be f32 or f16, got: ",
                    element_type);
    const auto& shape = mask_tensor->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, caller_name, ": Attention mask tensor rank must be >= 2, got shape: ", shape);

    const uint32_t row_dim = static_cast<uint32_t>(shape[shape.size() - 2]);
    const uint32_t col_dim = static_cast<uint32_t>(shape[shape.size() - 1]);
    OPENVINO_ASSERT(std::all_of(shape.begin(),
                                shape.end() - 2,
                                [](size_t dim) {
                                    return dim == 1;
                                }),
                    caller_name,
                    ": attention mask leading dimensions must be singleton, got shape: ",
                    shape);
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
    mv.element_type = element_type;
    mv.data = mask_tensor->data();
    return mv;
}

}  // namespace

void ov::npuw::util::write_swa_kv_slice_circular(ov::SoPtr<ov::ITensor> dst_tensor,
                                                 ov::SoPtr<ov::ITensor> src_new_kv,
                                                 uint32_t dst_kv_dim,
                                                 uint32_t src_kv_dim,
                                                 uint32_t num_stored_tokens_before,
                                                 uint32_t num_new_tokens) {
    // Circular SWA policy: token at absolute position p is stored at physical
    // slot (p % capacity), overwriting the oldest slot once the buffer is full.
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

namespace {

template <typename T>
void fill_causal_sliding_window_mask_typed(const MaskView& mask_view,
                                           uint32_t num_stored_tokens_before,
                                           uint32_t window_size) {
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

    const T kAttend = T(0.0f);
    const T kMasked = T(std::numeric_limits<ov::float16>::lowest());

    // Row columns = [past circular slots][current-chunk columns]
    //             = [0 .. past_width-1] [past_width .. past_width+row_dim-1].
    // past_width == 0 degenerates to current-chunk-only masking.
    //
    // Past slot -> absolute token index:
    //   unsaturated (stored_tokens_before < past_width): abs == slot, for slot < stored_tokens_before.
    //   saturated (stored_tokens_before >= past_width), wrap_slot = stored_tokens_before % past_width:
    //     [0, wrap_slot):          abs = (stored_tokens_before - wrap_slot) + slot
    //     [wrap_slot, past_width): abs = (stored_tokens_before - wrap_slot) + slot - past_width
    //   example: past_width=8, stored_tokens_before=11, wrap_slot=3 -> slot->abs: [8,9,10,3,4,5,6,7]
    //
    // Visibility: attend(abs) iff abs <= row_abs_pos AND row_abs_pos - abs < window_size, i.e.
    // abs in [row_abs_pos - window_size + 1, row_abs_pos]. The current-chunk area is the same
    // window clipped to the causal diagonal: local_c in [max(row_pad, row-window_size+1), row].

    // Inclusive [begin, end] slot interval; begin > end means "empty".
    struct VisibleRange {
        int64_t begin;
        int64_t end;
    };
    constexpr VisibleRange kEmptyRange{1, 0};

    // Step 1 helper: pure range arithmetic, no memory access. Clamps the visibility window
    // [range_begin, range_end] against a region's own domain [domain_begin, domain_end].
    auto compute_visible_range =
        [](int64_t range_begin, int64_t range_end, int64_t domain_begin, int64_t domain_end) -> VisibleRange {
        return {std::max(range_begin, domain_begin), std::min(range_end, domain_end)};
    };

    // Step 2 helper: pure memory write, no range math. Fills an already-clamped range.
    auto fill_visible_range = [](T* ptr, VisibleRange range, T fill_value) {
        if (range.begin <= range.end) {
            std::fill_n(ptr + range.begin, static_cast<size_t>(range.end - range.begin + 1), fill_value);
        }
    };

    // Visible slot ranges in the past region for one row. A saturated ring wraps into at most
    // two contiguous ranges (returned as a fixed-size array to avoid a per-row allocation).
    auto compute_visible_past_slots = [&](int64_t min_visible_abs_pos,
                                          int64_t max_visible_abs_pos) -> std::array<VisibleRange, 2> {
        if (!has_past_region) {
            return {kEmptyRange, kEmptyRange};
        }
        if (is_past_saturated) {
            // Saturated ring: at most two contiguous slot ranges can be visible,
            // one in [wrap_slot, past_width) and one in [0, wrap_slot).
            const int64_t ring_base_abs = stored_tokens_before_i64 - static_cast<int64_t>(wrap_slot);
            const int64_t older_segment_bias = ring_base_abs - past_width_i64;  // abs = older_segment_bias + slot

            // Segment 1: c in [wrap_slot, past_width-1].
            const VisibleRange segment1 = compute_visible_range(min_visible_abs_pos - older_segment_bias,
                                                                max_visible_abs_pos - older_segment_bias,
                                                                static_cast<int64_t>(wrap_slot),
                                                                past_width_i64 - 1);
            // Segment 2: c in [0, wrap_slot-1], only when the ring actually wrapped.
            const VisibleRange segment2 = (wrap_slot > 0u) ? compute_visible_range(min_visible_abs_pos - ring_base_abs,
                                                                                   max_visible_abs_pos - ring_base_abs,
                                                                                   0,
                                                                                   static_cast<int64_t>(wrap_slot) - 1)
                                                           : kEmptyRange;
            return {segment1, segment2};
        }
        if (stored_tokens_before > 0u) {
            // Unsaturated prefix: slot index equals absolute position for valid slots.
            return {compute_visible_range(min_visible_abs_pos, max_visible_abs_pos, 0, stored_tokens_before_i64 - 1),
                    kEmptyRange};
        }
        return {kEmptyRange, kEmptyRange};
    };

    T* base = static_cast<T*>(mask_view.data);
    for (uint32_t row = 0; row < row_dim; ++row) {
        T* row_ptr = base + static_cast<size_t>(row) * mask_view.col_dim;

        const int64_t row_i64 = static_cast<int64_t>(row);
        const int64_t row_abs_pos = stored_tokens_before_i64 + (row_i64 - row_pad_i64);
        const int64_t min_visible_abs_pos = row_abs_pos - window_i64 + 1;
        const int64_t max_visible_abs_pos = row_abs_pos;

        // Step 1: compute which slots are visible in this row. Pure arithmetic only --
        // nothing is written to the mask buffer yet.
        const std::array<VisibleRange, 2> past_slots =
            compute_visible_past_slots(min_visible_abs_pos, max_visible_abs_pos);

        // Current chunk ("present") diagonal region [past_width, past_width + row_dim).
        // local_c must satisfy all constraints below:
        //   1) valid key in right-aligned chunk: local_c >= row_pad
        //   2) causal:                         local_c <= row
        //   3) window:                         row - local_c < window_size
        // => local_c in [max(row_pad, row-window+1), row].
        const VisibleRange present_segment = compute_visible_range(std::max(row_pad_i64, row_i64 - window_i64 + 1),
                                                                   row_i64,
                                                                   0,
                                                                   static_cast<int64_t>(row_dim) - 1);

        // Step 2: populate masked and attended values.
        // 2.1 init row: all masked.
        std::fill_n(row_ptr, mask_view.col_dim, kMasked);
        // 2.2 attend past in ranges (from step 1).
        fill_visible_range(row_ptr, past_slots[0], kAttend);
        fill_visible_range(row_ptr, past_slots[1], kAttend);
        // 2.3 attend present.
        fill_visible_range(row_ptr + past_width, present_segment, kAttend);
    }
}

}  // namespace

void ov::npuw::util::fill_causal_sliding_window_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                                     uint32_t num_stored_tokens_before,
                                                     uint32_t num_real_new_tokens,
                                                     uint32_t window_size) {
    const auto mask_view = get_mask_view(mask_tensor, num_real_new_tokens, "fill_causal_sliding_window_mask");
    OPENVINO_ASSERT(window_size > 0, "fill_causal_sliding_window_mask: window_size must be > 0");

    switch (mask_view.element_type) {
    case ov::element::f32:
        fill_causal_sliding_window_mask_typed<float>(mask_view, num_stored_tokens_before, window_size);
        break;
    case ov::element::f16:
        fill_causal_sliding_window_mask_typed<ov::float16>(mask_view, num_stored_tokens_before, window_size);
        break;
    default:
        OPENVINO_THROW("fill_causal_sliding_window_mask: unsupported mask element type ", mask_view.element_type);
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
    fill_causal_sliding_window_mask(mask_tensor, num_stored_tokens_before, num_real_new_tokens, window_size);
    if (token_type_ids_real != nullptr) {
        overlay_vision_bidirectional_mask(mask_tensor, token_type_ids_real, num_real_new_tokens);
    }
}

namespace {

template <typename T>
void overlay_vision_bidirectional_mask_typed(const MaskView& mask_view,
                                             const int64_t* token_type_ids_real,
                                             uint32_t num_real_new_tokens) {
    const T kAttend = T(0.0f);
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
    T* base = static_cast<T*>(mask_view.data);
    auto apply_vision_run = [&](uint32_t run_start, uint32_t run_end_exclusive) {
        const uint32_t run_length = run_end_exclusive - run_start;
        const uint32_t run_col_start = mask_view.past_width + mask_view.row_pad + run_start;
        for (uint32_t row_index = run_start; row_index < run_end_exclusive; ++row_index) {
            T* row_ptr = base + static_cast<size_t>(mask_view.row_pad + row_index) * mask_view.col_dim;
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

}  // namespace

void ov::npuw::util::overlay_vision_bidirectional_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                                       const int64_t* token_type_ids_real,
                                                       uint32_t num_real_new_tokens) {
    if (num_real_new_tokens == 0) {
        return;
    }
    OPENVINO_ASSERT(token_type_ids_real != nullptr,
                    "overlay_vision_bidirectional_mask: token_type_ids_real must not be null");

    const auto mask_view = get_mask_view(mask_tensor, num_real_new_tokens, "overlay_vision_bidirectional_mask");

    switch (mask_view.element_type) {
    case ov::element::f32:
        overlay_vision_bidirectional_mask_typed<float>(mask_view, token_type_ids_real, num_real_new_tokens);
        break;
    case ov::element::f16:
        overlay_vision_bidirectional_mask_typed<ov::float16>(mask_view, token_type_ids_real, num_real_new_tokens);
        break;
    default:
        OPENVINO_THROW("overlay_vision_bidirectional_mask: unsupported mask element type ", mask_view.element_type);
    }
}

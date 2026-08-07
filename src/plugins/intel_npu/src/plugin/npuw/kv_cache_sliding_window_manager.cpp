// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_sliding_window_manager.hpp"

#include <algorithm>

#include "infer_request_utils.hpp"
#include "logging.hpp"
#include "util.hpp"

void ov::npuw::util::write_kv_slice_sliding(ov::SoPtr<ov::ITensor> dst_tensor,
                                            ov::SoPtr<ov::ITensor> src_new_kv,
                                            uint32_t dst_kv_dim,
                                            uint32_t src_kv_dim,
                                            uint32_t num_stored_tokens_before,
                                            uint32_t num_new_tokens,
                                            SlidingBufferLayout layout) {
    const uint32_t capacity = static_cast<uint32_t>(dst_tensor->get_shape()[dst_kv_dim]);
    const uint32_t old_total = num_stored_tokens_before;
    const uint32_t new_total = old_total + num_new_tokens;
    const uint32_t old_valid = std::min(old_total, capacity);
    const uint32_t new_valid = std::min(new_total, capacity);

    // Clamp against the source's own length too: a source tensor may legitimately hold
    // fewer valid tokens than `num_new_tokens` claims (e.g. when re-using another
    // layer's already-capacity-limited past buffer as a source, see the header comment).
    const uint32_t src_len = static_cast<uint32_t>(src_new_kv->get_shape()[src_kv_dim]);
    const uint32_t tokens_to_write = std::min({num_new_tokens, new_valid, src_len});

    if (layout == SlidingBufferLayout::Circular) {
        // No shift, ever: token at absolute position p always lives at physical index
        // (p % capacity). See SlidingBufferLayout's doc comment in the header for why
        // this is safe. Skip the leading tokens of this call that would be immediately
        // overwritten later in the very same call (mirrors the LeftAligned clamp above).
        if (tokens_to_write == 0) {
            return;
        }
        const uint32_t first_new_abs_pos = num_stored_tokens_before + (num_new_tokens - tokens_to_write);
        const uint32_t dst_start = first_new_abs_pos % capacity;

        auto src_slice = (src_len > tokens_to_write)
                             ? ov::npuw::util::make_tensor_slice(src_new_kv, src_kv_dim, src_len - tokens_to_write, src_len)
                             : src_new_kv;

        if (dst_start + tokens_to_write <= capacity) {
            // Single contiguous write - also covers the not-yet-saturated warm-up
            // phase, where dst_start == first_new_abs_pos, i.e. a plain append.
            auto dst_slice =
                ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, dst_start, dst_start + tokens_to_write);
            ov::npuw::util::copy_tensor_by_dim(src_slice, dst_slice, src_kv_dim, dst_kv_dim);
        } else {
            // Wraps past the end of the buffer: split into two contiguous legs.
            const uint32_t first_leg_len = capacity - dst_start;
            const uint32_t second_leg_len = tokens_to_write - first_leg_len;

            auto src_first_leg = ov::npuw::util::make_tensor_slice(src_slice, src_kv_dim, 0u, first_leg_len);
            auto dst_first_leg = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, dst_start, capacity);
            ov::npuw::util::copy_tensor_by_dim(src_first_leg, dst_first_leg, src_kv_dim, dst_kv_dim);

            auto src_second_leg =
                ov::npuw::util::make_tensor_slice(src_slice, src_kv_dim, first_leg_len, tokens_to_write);
            auto dst_second_leg = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, 0u, second_leg_len);
            ov::npuw::util::copy_tensor_by_dim(src_second_leg, dst_second_leg, src_kv_dim, dst_kv_dim);
        }
        return;
    }

    const uint32_t keep = new_valid - tokens_to_write;
    const bool needs_shift = (keep > 0 && keep < old_valid);

    if (needs_shift && dst_kv_dim == 3u) {
        // Transposed-V layout (dst_kv_dim == 3): a partial-slice shift touches only
        // `old_valid` of the `capacity` columns, but a dim-3 slice of a [1,C,H,W] tensor
        // is non-contiguous, so both the read (old_tail->copy_to) and the write
        // (copy_tensor_by_dim -> copy_columns_by_row_chunks) legs degrade into C*H
        // individual small (per-token) memory transactions. When dst_tensor lives in
        // NPU-resident remote memory, per-transaction latency (not bytes moved)
        // dominates, and C*H can be in the thousands - this is the empirically
        // dominant cost of the sliding-window KV update (~600ms/step on real HW).
        //
        // Since the *whole* (unsliced) buffer is fully contiguous, round-trip it as a
        // single big contiguous transfer instead: one bulk device->CPU copy, a cheap
        // in-CPU-memory shift (regular DRAM, C*H iterations here are negligible), then
        // one bulk CPU->device copy back. This trades "only move what changed" for
        // "always move `capacity` columns, but in O(1) device-memory transactions".
        LOG_DEBUG("[SWA] Bulk-shifting KV buffer (dim=3): keeping last "
                  << keep << " of " << old_valid << " old token(s), capacity=" << capacity);
        auto whole_tmp = ov::npuw::util::allocMem(dst_tensor->get_element_type(), dst_tensor->get_shape(), "CPU", nullptr);
        dst_tensor->copy_to(whole_tmp._ptr);  // single bulk contiguous transfer

        auto old_tail_cpu = ov::npuw::util::make_tensor_slice(whole_tmp, dst_kv_dim, old_valid - keep, old_valid);
        auto shift_tmp = ov::npuw::util::allocMem(dst_tensor->get_element_type(), old_tail_cpu->get_shape(), "CPU", nullptr);
        old_tail_cpu->copy_to(shift_tmp._ptr);  // CPU-to-CPU, cheap regardless of iteration count
        auto dst_front_cpu = ov::npuw::util::make_tensor_slice(whole_tmp, dst_kv_dim, 0u, keep);
        ov::npuw::util::copy_tensor_by_dim(shift_tmp, dst_front_cpu, dst_kv_dim, dst_kv_dim);  // CPU-to-CPU

        if (tokens_to_write > 0) {
            auto src_slice = (src_len > tokens_to_write)
                                 ? ov::npuw::util::make_tensor_slice(src_new_kv, src_kv_dim, src_len - tokens_to_write, src_len)
                                 : src_new_kv;
            auto dst_back_cpu = ov::npuw::util::make_tensor_slice(whole_tmp, dst_kv_dim, keep, keep + tokens_to_write);
            ov::npuw::util::copy_tensor_by_dim(src_slice, dst_back_cpu, src_kv_dim, dst_kv_dim);
        }

        whole_tmp->copy_to(dst_tensor._ptr);  // single bulk contiguous transfer back
        return;
    }

    if (needs_shift) {
        // Sliding window is (re)saturated: shift the surviving tail of the old content to
        // the front of the buffer. A temporary CPU snapshot is used because dst and the
        // "old" region alias the same tensor, making a direct in-place copy unsafe.
        LOG_DEBUG("[SWA] Shifting KV buffer: keeping last " << keep << " of " << old_valid << " old token(s), dim="
                                                             << dst_kv_dim << ", capacity=" << capacity);
        auto old_tail = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, old_valid - keep, old_valid);
        auto tmp = ov::npuw::util::allocMem(dst_tensor->get_element_type(), old_tail->get_shape(), "CPU", nullptr);
        old_tail->copy_to(tmp._ptr);
        auto dst_front = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, 0u, keep);
        ov::npuw::util::copy_tensor_by_dim(tmp, dst_front, dst_kv_dim, dst_kv_dim);
    }

    if (tokens_to_write == 0) {
        return;
    }
    auto src_slice = (src_len > tokens_to_write)
                         ? ov::npuw::util::make_tensor_slice(src_new_kv, src_kv_dim, src_len - tokens_to_write, src_len)
                         : src_new_kv;
    auto dst_back = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, keep, keep + tokens_to_write);
    ov::npuw::util::copy_tensor_by_dim(src_slice, dst_back, src_kv_dim, dst_kv_dim);
}

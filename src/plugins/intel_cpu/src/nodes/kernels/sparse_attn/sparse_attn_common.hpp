// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>

#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/softmax_kernel.hpp"
#include "onednn/iml_type_mapper.h"
#include "utils/general_utils.h"

namespace ov::Extensions::Cpu {

/// @brief Query heads that share both K/V rows and the selection, processed together by one work item:
/// gcd(q_head_num / k_head_num, q_head_num / sel_head_num). A selection head count of 1 or k_head_num gives the whole
/// K-head group, q_head_num gives 1.
inline size_t max_q_head_pack_size(size_t q_head_num, size_t k_head_num, size_t sel_head_num) {
    return std::gcd(q_head_num / k_head_num, q_head_num / sel_head_num);
}

}  // namespace ov::Extensions::Cpu

// Everything below is used by the cross-compiled kernels only. It lives in XARCH so that every ISA build has its own
// copy: an inline function in a shared namespace would be merged by the linker, and the AVX-512 copy could then run on
// an AVX2 CPU.
namespace ov::Extensions::Cpu::XARCH {

/// @brief The implementation type of the SIMD stages of this instruction-set variant. It is fixed when the variant is
/// compiled, as the dispatcher picks the variant by the same CPU check.
inline ov::intel_cpu::impl_desc_type variant_impl_type() {
#if defined(HAVE_AVX512F)
    return ov::intel_cpu::impl_desc_type::avx512;
#elif defined(HAVE_AVX2)
    return ov::intel_cpu::impl_desc_type::avx2;
#else
    return ov::intel_cpu::impl_desc_type::ref;
#endif
}

/// @brief Lays out buffers one after the other, each aligned to 64 bytes, so a scratch's size and the places of its
/// buffers come from the same code: with a null base it only measures (bytes()), with a base take() returns the
/// buffers, reserve() the offsets.
class ScratchPartition {
public:
    static constexpr size_t kAlign = 64;

    explicit ScratchPartition(uint8_t* base = nullptr) : m_base(base) {}

    /// @brief The offset of the next buffer of `bytes` bytes.
    size_t reserve(size_t bytes) {
        const size_t start = m_bytes;
        m_bytes = ov::intel_cpu::rnd_up(m_bytes + bytes, kAlign);
        return start;
    }
    template <typename T>
    T* take(size_t count) {
        const size_t start = reserve(count * sizeof(T));
        return m_base != nullptr ? reinterpret_cast<T*>(m_base + start) : nullptr;
    }
    [[nodiscard]] size_t bytes() const {
        return m_bytes;
    }

private:
    uint8_t* m_base;
    size_t m_bytes = 0;
};

/// @brief Number of rows of [kpos0, kpos0 + n) a query may attend, up to `valid_end` (exclusive); 0 when the range
/// starts before `first_valid` (the start of the real history of a left-padded sequence: callers start at or after it).
inline int64_t visible_rows(int64_t kpos0, int64_t n, int64_t first_valid, int64_t valid_end) {
    return kpos0 < first_valid ? 0 : std::max<int64_t>(std::min(n, valid_end - kpos0), 0);
}

/// @brief First position of the causal tail of the query at `token_pos`: its incomplete last block, counted from `pad`
/// (the physical start of the history); empty at a block boundary. sel_indices / sel_count hold complete blocks only,
/// so causal consumers attend the tail themselves.
inline int64_t causal_tail_start(int64_t token_pos, int64_t pad, int64_t key_block_size) {
    return pad + ((token_pos - pad + 1) / key_block_size) * key_block_size;
}

/// @brief KV chunks the selected blocks of each SIMD item are split into, so few (batch, head pack, query) items
/// still fill the threads; the partial softmax states are merged afterwards. 1 = no split (enough items, or too few
/// blocks).
inline size_t kv_chunk_count(size_t items, size_t max_blocks, size_t num_threads) {
    constexpr size_t kMinKvChunkBlocks = 16;
    if (items == 0 || items >= num_threads) {
        return 1;
    }
    const size_t kv_chunks = std::min(num_threads / items, max_blocks / kMinKvChunkBlocks);
    return kv_chunks > 1 ? kv_chunks : 1;
}

/// @brief Merges the partial online-softmax states of one pack's `kv_chunks` slices into the normalized outputs
/// `out[pack, Dv]` (f32). A state is run_max[pack] | run_sum[pack] | acc[pack, Dv], laid out one after the other. A
/// head without attended rows gets zeros.
inline void merge_kv_chunk_partials(const float* partials, size_t kv_chunks, size_t pack, size_t Dv, float* out) {
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    const size_t slot = pack * (Dv + 2);
    for (size_t head = 0; head < pack; head++) {
        float max = kNegInf;
        for (size_t c = 0; c < kv_chunks; c++) {
            max = std::max(max, partials[c * slot + head]);
        }
        float* acc = out + head * Dv;
        std::fill(acc, acc + Dv, 0.0F);
        float sum = 0.0F;
        if (max > kNegInf) {
            for (size_t c = 0; c < kv_chunks; c++) {
                const float* part = partials + c * slot;
                if (part[head] == kNegInf) {
                    continue;
                }
                const float weight = std::exp(part[head] - max);
                sum += weight * part[pack + head];
                const float* part_acc = part + 2 * pack + head * Dv;
                for (size_t d = 0; d < Dv; d++) {
                    acc[d] += weight * part_acc[d];
                }
            }
        }
        const float inv_sum = sum > 0.0F ? 1.0F / sum : 0.0F;
        for (size_t d = 0; d < Dv; d++) {
            acc[d] *= inv_sum;
        }
    }
}

/// @brief Starts the attention of a head pack: zero accumulators, empty online softmax.
inline void init_pack_state(float* acc, float* running_max, float* running_sum, size_t pack_size, size_t v_head_size) {
    std::fill_n(acc, pack_size * v_head_size, 0.0F);
    std::fill_n(running_max, pack_size, -std::numeric_limits<float>::infinity());
    std::fill_n(running_sum, pack_size, 0.0F);
}

/// @brief Online-softmax step of one head over scores[0, rows): rescales the running state (maximum, sum and the
/// accumulator acc[v_head_size]) to the new maximum and turns the scores into weights in place.
inline void online_softmax_step(float* scores,
                                size_t rows,
                                float scale,
                                float& running_max,
                                float& running_sum,
                                float* acc,
                                size_t v_head_size) {
    float scaled_max = std::numeric_limits<float>::lowest();
    scale_add2_reduce_max<false, false, false, false, float>(scores,
                                                             scale,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             false,
                                                             rows,
                                                             0.0F,
                                                             scaled_max,
                                                             nullptr,
                                                             0);
    const float new_max = std::max(running_max, scaled_max);
    if (new_max > running_max) {
        // corr == 0 for the first call (running_max == -inf), where acc and running_sum are still zero
        const float corr = std::exp(running_max - new_max);
        running_sum *= corr;
        for (size_t d = 0; d < v_head_size; d++) {
            acc[d] *= corr;
        }
        running_max = new_max;
    }
    float sum = 0.0F;
    exp_reduce_sum(scores, new_max, rows, sum);
    running_sum += sum;
}

/// @brief Ends the attention of a head pack: divides every head's accumulator by its softmax sum (0 when nothing was
/// visible) and stores it as T to out_row(head), the row of v_head_size elements of the output.
template <typename OutRow>
void store_normalized_pack(float* acc,
                           const float* running_sum,
                           size_t pack_size,
                           size_t v_head_size,
                           const OutRow& out_row) {
    for (size_t head = 0; head < pack_size; head++) {
        float* head_acc = acc + head * v_head_size;
        const float inv_sum = running_sum[head] > 0.0F ? 1.0F / running_sum[head] : 0.0F;
        for (size_t d = 0; d < v_head_size; d++) {
            head_acc[d] *= inv_sum;
        }
        cvt_copy(out_row(head), head_acc, 1, v_head_size, 0, 0);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH

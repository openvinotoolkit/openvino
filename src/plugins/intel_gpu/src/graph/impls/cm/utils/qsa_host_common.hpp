// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Host-side helpers shared by the QSA indexer / sparse-attention CM implementations
// (dense and paged variants differ only in KV addressing, not in these rules).

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string_view>

#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::intel_gpu::cm {

/// The CM implementation is opt-in: it is selected only when OV_GPU_QSA_IMPL=cm.
inline bool cm_impl_allowed() {
    const char* v = std::getenv("OV_GPU_QSA_IMPL");
    return v != nullptr && std::string_view(v) == "cm";
}

inline constexpr const char* qsa_cm_build_options = " -cmc -Qxcm_register_file_size=256";

/// Appends a 32-bit kernel scalar; kernels index with uint32, so wrapping would corrupt addressing.
inline void push_u32(KernelData& kd, size_t value) {
    OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU][CM] QSA kernel scalar ", value, " does not fit in 32 bits");
    cldnn::scalar_desc s;
    s.t = cldnn::scalar_desc::Types::UINT32;
    s.v.u32 = static_cast<uint32_t>(value);
    kd.params.scalars.push_back(s);
}

/// Upper bound on the decode BLOCK_SCORES scratch. Debug builds can shrink it through
/// OV_GPU_QSA_SCORE_BUDGET_BYTES to force the multi-chunk path with small shapes.
inline size_t score_budget_bytes() {
#ifdef GPU_DEBUG_CONFIG
    if (const char* override_str = std::getenv("OV_GPU_QSA_SCORE_BUDGET_BYTES")) {
        const auto value = std::strtoull(override_str, nullptr, 10);
        if (value > 0) {
            return static_cast<size_t>(value);
        }
    }
#endif
    return 128 * 1024 * 1024;
}

/// Cooperative top-k work-group size: decode rows span the whole history, while prefill
/// rows are short and numerous, so smaller work-groups pack the GPU better.
constexpr size_t topk_wg_size(bool decode) {
    return decode ? 16 : 8;
}

// ---------------------------------------------------------------------------
// Sparse-attention DPAS routing. `Desc` is cldnn::sparse_sdpa or cldnn::sparse_pa; both
// expose num_heads/num_kv_heads/k_head_size/v_head_size/key_block_size/causal.
// `page_size` is std::nullopt for dense K/V, where no page-boundary constraint exists.
// ---------------------------------------------------------------------------
inline constexpr size_t dpas_query_tile = 16;  // dpas N axis == GRF width on Xe2
inline constexpr size_t dpas_k_step = 16;      // fp16 K elements per dpas
inline constexpr size_t dpas_wide_cols = 32;   // halfs per 2D block row (64-byte hardware cap)
inline constexpr size_t gqa_workers = 4;       // GQA_W in the *_gqa_dpas.cm kernels

// Decode-only split-selected-KV attention: partitions each token's selected-block list
// across independent producers to keep the GPU occupied for long-history decode.
inline constexpr size_t split_partitions = 16;
// One GQA-DPAS WG serves all n_rep heads of a KV head, so decode needs more WGs per row.
inline constexpr size_t gqa_split_partitions = 64;

/// Dv slice merged by one split_reduce thread.
inline size_t split_reduce_cols(size_t v_head_size) {
    return v_head_size % 32 == 0 ? 32 : 16;
}

/// Heads that share one dpas tile: filling the N axis with HT heads x (16/HT) tokens shrinks
/// the per-tile top-k union from 16 independent selections down to 16/HT.
template <class Desc>
size_t dpas_heads_per_tile(const Desc& desc) {
    const auto n_rep = desc.num_heads / desc.num_kv_heads;
    // 16 heads would leave one token per tile, and load_vnni_cols<1> does not compile.
    for (size_t ht : {size_t{8}, size_t{4}, size_t{2}}) {
        if (n_rep % ht == 0) {
            return ht;
        }
    }
    return 1;
}

/// Threads that cooperate on one (query tile, head), bounded by a register budget.
/// Returns 0 when no supported split fits, which disables the systolic path.
template <class Desc>
size_t dpas_attention_workers(const Desc& desc) {
    constexpr size_t reg_budget = 12 * 1024;
    for (size_t w : {size_t{1}, size_t{2}, size_t{4}, size_t{8}}) {
        if (desc.k_head_size % (w * dpas_k_step) != 0 || desc.v_head_size % (w * dpas_query_tile) != 0) {
            continue;
        }
        const auto dh_w = desc.k_head_size / w;
        const auto dv_w = desc.v_head_size / w;
        if (dh_w % dpas_wide_cols != 0 || dv_w % dpas_wide_cols != 0) {
            continue;
        }
        const auto rq = dh_w * dpas_query_tile * sizeof(ov::float16);
        const auto ro = dpas_query_tile * dv_w * sizeof(float);
        const auto kv = dpas_k_step * (dh_w + dv_w) * sizeof(ov::float16);
        if (rq + ro + kv <= reg_budget) {
            return w;
        }
    }
    return 0;
}

/// Whether the plain systolic attention kernel can be used.
template <class Desc>
bool can_use_dpas_attention(const Desc& desc, std::optional<size_t> page_size, const cldnn::device_info& info) {
    if (info.arch < cldnn::gpu_arch::xe2) {
        return false;
    }
    if (desc.k_head_size % dpas_k_step != 0 || desc.v_head_size % dpas_query_tile != 0) {
        return false;
    }
    if (desc.k_head_size * sizeof(ov::float16) < 64 || desc.v_head_size * sizeof(ov::float16) < 64) {
        return false;
    }
    if (desc.key_block_size % 2 != 0 || dpas_k_step % desc.key_block_size != 0) {
        return false;  // four QSA blocks must pack exactly one 16-token dpas K step
    }
    if (page_size && *page_size % desc.key_block_size != 0) {
        return false;  // a compressed block must not cross a KV page
    }
    return dpas_attention_workers(desc) != 0;
}

/// Whether the GQA-aware systolic attention kernel can be used; preferred over the plain
/// one whenever it holds (one KV read serves the whole n_rep group in one WG).
template <class Desc>
bool can_use_gqa_dpas_attention(const Desc& desc, std::optional<size_t> page_size, const cldnn::device_info& info) {
    if (info.arch < cldnn::gpu_arch::xe2) {
        return false;
    }
    if (desc.num_kv_heads == 0 || desc.num_heads % desc.num_kv_heads != 0) {
        return false;
    }
    const auto n_rep = desc.num_heads / desc.num_kv_heads;
    if (n_rep < 1 || n_rep > 16) {
        return false;
    }
    if (desc.k_head_size % (gqa_workers * 32) != 0 || desc.v_head_size % (gqa_workers * 32) != 0) {
        return false;
    }
    if (std::max(desc.k_head_size, desc.v_head_size) > 256) {
        return false;  // above the verified accumulator range
    }
    if (desc.key_block_size != 2 && desc.key_block_size != 4 && desc.key_block_size != 8 && desc.key_block_size != 16) {
        return false;
    }
    if (page_size && (*page_size == 0 || *page_size % desc.key_block_size != 0)) {
        return false;
    }
    return desc.causal;
}

/// Partitions behind SPLIT_ACC/SPLIT_ML and split_reduce for this config's decode producer.
template <class Desc>
size_t decode_split_partitions(const Desc& desc, std::optional<size_t> page_size, const cldnn::device_info& info) {
    return can_use_gqa_dpas_attention(desc, page_size, info) ? gqa_split_partitions : split_partitions;
}

// Dense overloads (no paging).
template <class Desc>
bool can_use_dpas_attention(const Desc& desc, const cldnn::device_info& info) {
    return can_use_dpas_attention(desc, std::nullopt, info);
}
template <class Desc>
bool can_use_gqa_dpas_attention(const Desc& desc, const cldnn::device_info& info) {
    return can_use_gqa_dpas_attention(desc, std::nullopt, info);
}
template <class Desc>
size_t decode_split_partitions(const Desc& desc, const cldnn::device_info& info) {
    return decode_split_partitions(desc, std::nullopt, info);
}

}  // namespace ov::intel_gpu::cm

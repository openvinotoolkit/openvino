// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <utility>

#include "../ocl_v2/utils/jitter.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/type.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {
constexpr int32_t PA_CM_REGISTER_FILE_SIZE = 256;

// This function returns the kv_step and kv_split_len based on the architecture.
// return {kv_step, kv_split_len}
inline std::pair<size_t, size_t> get_kv_split_size(size_t arch) {
    if (arch == 1) {
        return {8, 32};  // For Xe1
    } else if (arch == 2) {
        return {16, 32};  // For Xe2
    }
    OPENVINO_ASSERT(false, "Unsupported architecture for KV split size");
    return {0, 0};  // Fallback case, should not be reached
}

struct SingleTokenQChunking {
    int32_t q_head_chunks_per_kv_head;
    int32_t q_head_chunk_size;
};

inline SingleTokenQChunking get_single_token_q_chunking(const kernel_impl_params& params, const paged_attention& desc, size_t kv_partition_size) {
    // Must match kernel mapping in pa_single_token.cm:
    //   kv_head_num_idx = gid1 / Q_head_chunks_per_kv_head
    //   head_num_idx    = gid1 * Q_head_chunk_size
    // Kernel does not guard extra heads, so we must ensure exact coverage:
    //   Q_head_chunks_per_kv_head * Q_head_chunk_size == q_heads_per_kv_head
    constexpr int32_t MaxRepeatCount = 8;

    const auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    const int32_t q_heads_per_kv_head = static_cast<int32_t>(desc.heads_num / desc.kv_heads_num);

    // Match kernel arch-dependent params
    const int32_t reg_n = (xe_arch == 1) ? 8 : 16;
    const int32_t kv_step = static_cast<int32_t>(get_kv_split_size(xe_arch).first);
    constexpr int32_t reg_m = 1;  // RepeatCount
    constexpr int32_t bytes_per_float = 4;

    const int32_t kv_partition_step_num = static_cast<int32_t>(kv_partition_size / kv_step);
    const int32_t rs_cols = reg_m * kv_partition_step_num * reg_n;

    const int32_t reg_file_size = PA_CM_REGISTER_FILE_SIZE;
    const int32_t grf_bytes = (xe_arch == 1) ? 32 : 64;
    const int32_t budget_bytes = reg_file_size * grf_bytes - 1;

    int32_t max_q_by_matrix = budget_bytes / (bytes_per_float * rs_cols);
    if (max_q_by_matrix < 1)
        max_q_by_matrix = 1;

    const int32_t target_chunk = std::min<int32_t>(MaxRepeatCount, max_q_by_matrix);

    int32_t q_head_chunk_size = std::min<int32_t>(q_heads_per_kv_head, target_chunk);
    while (q_head_chunk_size > 1 && (q_heads_per_kv_head % q_head_chunk_size) != 0) {
        --q_head_chunk_size;
    }
    const int32_t q_head_chunks_per_kv_head = q_heads_per_kv_head / q_head_chunk_size;

    return {q_head_chunks_per_kv_head, q_head_chunk_size};
}

inline std::string get_pa_build_options() {
    return " -cmc -Qxcm_register_file_size=" + std::to_string(PA_CM_REGISTER_FILE_SIZE);
}

#define FIND_DEBUG_ACC 0
// The block size for KV cache is set to 256 for xattn to achieve better performance.
// For non-xattn case, it can be set to 16 for compatibility to legacy implementations.
#define PA_KV_CACHE_BLOCK_SIZE_LEGACY 16
#define PA_KV_CACHE_BLOCK_SIZE_XATTN  256

constexpr uint32_t SG_M = 4;
constexpr uint32_t SG_N = 8;
constexpr int STRIDE = 16;

enum class PagedAttentionStage : uint8_t { GENERATE = 0, PREFILL = 1, MIXED = 2, UNKNOWN = 3 };
enum class MixedRouteMode : uint8_t { MULTI = 0, SPLIT = 1 };
struct PagedAttentionRuntimeParams : public ImplRuntimeParams {
    // common runtime state
    PagedAttentionStage stage;       // Current PA execution stage
    size_t max_context_len;          // Maximum KV context length in current batch
    size_t batch_size_in_sequences;  // Number of subsequences in current request

    // single-token/decode path
    size_t num_of_partitions;                // Number of KV partitions for decode/finalization
    SingleTokenQChunking q_chunking;         // Cached single-token Q-head chunking parameters
    size_t single_token_selected_count = 0;  // Number of subsequences routed to single-token kernel

    // multi-token dispatch size
    size_t multi_token_wg_count = 0;  // Number of WGs required by pa_multi_token

    // xattention runtime state
    bool enable_xattn_estimation = false;  // Whether xattn estimate stages are enabled
    size_t xattn_block_size = 1;           // Selected xattn sparse block size (1/128/256)
    size_t xattn_num_subseqs = 1;          // Number of subsequences participating in xattn path

    // xattention dispatch sizes
    size_t xattn_gemmqk_wg_count = 0;  // Exact WG count for xattn_gemm_qk
    size_t xattn_find_wg_count = 0;    // Exact WG count for xattn_find_block
    size_t xattn_post_wg_count = 0;    // Exact WG count for xattn_post_proc

    // xattention internal buffer sizing
    size_t xattn_cumul_kq_max_bytes = 0;   // Total bytes for XATTN_GEMMQK_MAX
    size_t xattn_cumul_exp_sum_bytes = 0;  // Total bytes for XATTN_GEMMQK_EXPSUMS
    size_t xattn_cumul_mask_elems = 0;     // Total elements for XATTN_BLOCKMASK
    size_t xattn_cumul_mask_wg_elems = 0;  // Total elements for XATTN_BLOCKMASK_MERGED
    size_t xattn_meta_num_int32s = 0;      // Total int32 count in XATTN_SUBSEQ_META
};

enum PagedAttentionInternBuffIdx {
    // Decode scratch buffers used by generate path and split-mixed single-token path.
    DECODE_PARTITIONOUT = 0,  // 0: f32 partial attention outputs before final reduction
    DECODE_EXPSUMS = 1,       // 1: f32 softmax exp-sum accumulators for partition reduction

    // Routing scratch buffers used to map subsequences onto decode/multi-token kernels.
    MULTI_TOKEN_WG_MAPPING = 2,         // 2: i32 pairs [block_start_pos, subsequence_id]
    SINGLE_TOKEN_SELECTED_SEQ_IDS = 3,  // 3: i32 subsequence ids selected for single-token dispatch

    // XAttention estimate scratch buffers for multi-token sparse-attention path.
    XATTN_GEMMQK_MAX = 4,        // 4: f32 max logits per GEMM-QK work-group tile
    XATTN_GEMMQK_EXPSUMS = 5,    // 5: f32 partial exp-sums produced by GEMM-QK stage
    XATTN_BLOCKMASK = 6,         // 6: boolean sparse block mask per q-block / k-block pair
    XATTN_BLOCKMASK_MERGED = 7,  // 7: boolean sparse block mask after q-block merge in post-proc
    XATTN_SUBSEQ_META = 8,       // 8: i32 per-subsequence metadata table (16 entries per subsequence)
    XATTN_FIND_WG_MAP = 9,       // 9: i32 pairs [subseq_id, q_block_idx] for find-block dispatch
    XATTN_POST_WG_MAP = 10,      // 10: i32 pairs [subseq_id, merged_q_block_idx] for post-proc dispatch
#if FIND_DEBUG_ACC
    XATTN_FIND_DEBUG_ACC = 11,  // 11: f16 debug-only KQ accumulation buffer
#endif
};

//-----------------------------------------------------------------------------------------------------------------
// Helpers of XAttention
//-----------------------------------------------------------------------------------------------------------------
// Stage/context helpers shared across CM paged-attention implementation units.
PagedAttentionStage get_paged_attention_stage(const kernel_impl_params& impl_param);
size_t get_max_context_len(const kernel_impl_params& params);
size_t get_batch_size_in_sequences(const std::vector<layout>& input_layouts);

// XAttention policy helpers.
float get_xattn_thresh(const kernel_impl_params& impl_param, const size_t seq_idx = 0);
bool bypass_xattn(const kernel_impl_params& impl_param);

class PagedAttentionGeneratorBase : public KernelGenerator {
public:
    explicit PagedAttentionGeneratorBase(std::string_view kernel_name, std::string_view stage_suffix = "_cm") : KernelGenerator(kernel_name, stage_suffix) {}
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_pa_build_options();
    }
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
};

class PagedAttentionGeneratorKVCacheUpdate : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorKVCacheUpdate() : PagedAttentionGeneratorBase("pa_kv_cache_update_ref") {}
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

private:
    static size_t get_kv_update_wg_size(const RuntimeParams& params);
};

class PagedAttentionGeneratorMultiToken : public PagedAttentionGeneratorBase {
public:
    static constexpr size_t _wg_size = 16;

    explicit PagedAttentionGeneratorMultiToken(size_t xattn_block_size = 1)
        : PagedAttentionGeneratorBase("pa_multi_token", "_cm_bs" + std::to_string(xattn_block_size)),
          _xattn_block_size(xattn_block_size) {}

    static size_t get_q_step(const kernel_impl_params& params) {
        const auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
        if (xe_arch == 1) {
            return 8;  // For Xe1
        }
        // For Xe2, q_step = CM_GRF_WIDTH / 32
        return 16;  // For Xe2+
    }

    static size_t get_wg_seq_len(const kernel_impl_params& params) {
        return _wg_size * get_q_step(params);
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

private:
    size_t _xattn_block_size;
};

class PagedAttentionGeneratorSingleToken : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorSingleToken() : PagedAttentionGeneratorBase("pa_single_token") {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

    static size_t get_partition_size(const bool has_xattention = false) {
        // inheristic setting for single token to ensure the best performance, which is also verified by
        // internal testing. We can consider to make it configurable if needed in the future.
        if (!has_xattention && PA_KV_CACHE_BLOCK_SIZE_LEGACY < 128) {
            return 128;
        } else {
            return PA_KV_CACHE_BLOCK_SIZE_XATTN;
        }
    }
};

class PagedAttentionGeneratorSingleTokenFinalization : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorSingleTokenFinalization() : PagedAttentionGeneratorBase("pa_single_token_finalization") {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

//-----------------------------------------------------------------------------------------------------------------
// XAttention Estimate generators
//-----------------------------------------------------------------------------------------------------------------
class XAttentionEstimateGeneratorBase : public KernelGenerator {
public:
    explicit XAttentionEstimateGeneratorBase(std::string_view kernel_name, size_t xattn_block_size)
        : KernelGenerator(kernel_name, "_cm_bs" + std::to_string(xattn_block_size)),
          _xattn_block_size(xattn_block_size) {}
    static uint32_t get_block_sg_m(const kernel_impl_params& params) {
        return is_xe2_or_xe3(params) ? 64u : 32u;
    }

    static uint32_t get_block_sg_n(const kernel_impl_params& params) {
        return is_xe2_or_xe3(params) ? 32u : 16u;
    }

    static uint32_t get_block_wg_m(const kernel_impl_params& params) {
        return get_block_sg_m(params) * SG_M;
    }

    static uint32_t get_block_wg_n(const kernel_impl_params& params) {
        return get_block_sg_n(params) * SG_N;
    }

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_pa_build_options();
    }
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;

protected:
    size_t _xattn_block_size;

private:
    static bool is_xe2_or_xe3(const kernel_impl_params& params) {
        const auto arch = params.get_device_info().arch;
        return arch == gpu_arch::xe2 || arch == gpu_arch::xe3;
    }
};
class XAttentionEstimateGEMMQK : public XAttentionEstimateGeneratorBase {
public:
    explicit XAttentionEstimateGEMMQK(size_t xattn_block_size) : XAttentionEstimateGeneratorBase("xattn_gemm_qk", xattn_block_size) {}
    XAttentionEstimateGEMMQK() = delete;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

class XAttentionEstimateFindBlock : public XAttentionEstimateGeneratorBase {
public:
    explicit XAttentionEstimateFindBlock(size_t xattn_block_size) : XAttentionEstimateGeneratorBase("xattn_find_block", xattn_block_size) {}
    XAttentionEstimateFindBlock() = delete;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

class XAttentionEstimatePostProc : public XAttentionEstimateGeneratorBase {
public:
    explicit XAttentionEstimatePostProc(size_t xattn_block_size) : XAttentionEstimateGeneratorBase("xattn_post_proc", xattn_block_size) {}
    XAttentionEstimatePostProc() = delete;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

}  // namespace ov::intel_gpu::cm
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
constexpr size_t WG_SIZE = 16;
constexpr int STRIDE = 16;

enum class PagedAttentionStage : uint8_t { GENERATE = 0, PREFILL = 1, MIXED = 2, UNKNOWN = 3 };
struct PagedAttentionRuntimeParams : public ImplRuntimeParams {
    PagedAttentionStage stage;
    size_t max_context_len;
    // below are rt params for decoding
    size_t num_of_partitions;
    // cached single-token Q chunking
    SingleTokenQChunking q_chunking;
    // below are rt params for xattn
    size_t block_wg_m;
    size_t q_block_pad;
    size_t k_block_pad;
    size_t q_stride_pad;
    size_t q_block_pad_merged;
    size_t N_kq_groups;
    size_t M;
    size_t N;
    size_t K;
    size_t xattn_block_size;
};

enum PagedAttentionInternBuffIdx {
    // for decoding kernels
    DECODE_PARTITIONOUT = 0,  // 0: intermediate partition output
    DECODE_EXPSUMS = 1,       // 1: softmax exp_sums
    // for xattn kernels
    XATTN_GEMMQK_MAX = 2,        // 2: kq_max_wg
    XATTN_GEMMQK_EXPSUMS = 3,    // 3: kq_exp_partial_sum
    XATTN_BLOCKMASK = 4,         // 4: sparse_block_mask
    XATTN_BLOCKMASK_MERGED = 5,  // 5: sparse_block_mask_wg
#if FIND_DEBUG_ACC
    XATTN_FIND_DEBUG_ACC = 6,  // 6: kq_sum for debug purpose only
    TQ_Q_TRANSFORM = 7,        // 7: TurboQuant query rotation transform (q_t)
    TQ_CENTROIDS = 8,          // 8: TurboQuant centroids LUT
    TQ_BOUNDARIES = 9,         // 9: TurboQuant boundaries
#else
    TQ_Q_TRANSFORM = 6,  // 6: TurboQuant query rotation transform (q_t)
    TQ_CENTROIDS = 7,    // 7: TurboQuant centroids LUT
    TQ_BOUNDARIES = 8,   // 8: TurboQuant boundaries
#endif
};

//-----------------------------------------------------------------------------------------------------------------
// Helpers of XAttention
//-----------------------------------------------------------------------------------------------------------------
int64_t get_aligned_seq_len(const kernel_impl_params& impl_param, const PagedAttentionStage& stage, int64_t target_seq_len_block_size);
PagedAttentionStage get_paged_attention_stage(const kernel_impl_params& impl_param);
size_t get_max_context_len(const kernel_impl_params& params);
size_t get_past_len(const kernel_impl_params& params, const size_t seq_idx);

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
    explicit PagedAttentionGeneratorKVCacheUpdate(bool turboquant = false)
        : PagedAttentionGeneratorBase(turboquant ? "compressed_kv_cache_update_tq" : "pa_kv_cache_update_ref"),
          _turboquant(turboquant) {}

    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

private:
    bool _turboquant = false;
};

class PagedAttentionGeneratorMultiToken : public PagedAttentionGeneratorBase {
public:
    explicit PagedAttentionGeneratorMultiToken(size_t xattn_block_size = 1, bool turboquant = false)
        : PagedAttentionGeneratorBase(turboquant ? "pa_multi_token_turboquant" : "pa_multi_token",
                                     turboquant ? "" : "_cm_bs" + std::to_string(xattn_block_size)),
          _xattn_block_size(xattn_block_size),
          _turboquant(turboquant) {}

    static size_t get_q_step(const kernel_impl_params& params) {
        const auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
        if (xe_arch == 1) {
            return 8;  // For Xe1
        }
        // For Xe2, q_step = CM_GRF_WIDTH / 32
        return 16;  // For Xe2+
    }

    static size_t get_wg_seq_len(const kernel_impl_params& params) {
        return WG_SIZE * get_q_step(params);
    }

    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

private:
    size_t _xattn_block_size;
    bool _turboquant = false;
};

class PagedAttentionGeneratorSingleToken : public PagedAttentionGeneratorBase {
public:
    explicit PagedAttentionGeneratorSingleToken(bool turboquant = false)
        : PagedAttentionGeneratorBase(turboquant ? "pa_single_token_turboquant" : "pa_single_token"),
          _turboquant(turboquant) {}

    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override;
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

private:
    bool _turboquant = false;
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
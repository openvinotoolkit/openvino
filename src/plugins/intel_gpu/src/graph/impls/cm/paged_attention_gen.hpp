// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <array>
#include <utility>
#include <vector>

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
    }
    if (arch == 2) {
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

// Subsequences with q_len in (1, SMALL_Q_THRESHOLD] and past_len > 0 are routed
// to pa_small_q in split-mixed mode. Bound matches the typical EAGLE/draft
// spec_num = 16 and the Q_head_chunk_size × KV_PARTITION_STEP_NUM register
// budget that fits the rS tile under -Qxcm_register_file_size.
constexpr size_t SMALL_Q_THRESHOLD = 16;

// TILE_Q: how many q-tokens one pa_small_q *workgroup* packs into the DPAS M dim.
//
// Q_ROWS = Q_head_chunk_size * TILE_Q is the workgroup's total q-row count. It is no longer
// bounded by the DPAS RepeatCount cap of 8: the kernel splits Q_ROWS across
// WG_THREADS = Q_ROWS / 8 threads of one workgroup, each issuing RepeatCount 8, and those
// threads share the dequantized/transposed K/V tile through SLM. That sharing is the whole
// point -- the marshal and the K/V traffic are per *tile*, not per q-row -- so a bigger
// TILE_Q amortises the dominant cost even when it wastes q-rows.
//
// Concretely at spec_num=16, GQA-4: TILE_Q=16 gives one workgroup of 8 threads that marshals
// the KV partition once, where TILE_Q=2 gave 8 independent single-thread workgroups each
// marshalling the whole partition. The extra DPAS work from padding a short spec window up
// to TILE_Q is cheap by comparison (DPAS is ~8 % of the kernel); the marshal is not.
//
// That said, padding is not free either, which is what get_small_q_tile_q_alt below exists
// for. A row past valid_count is not an idle thread: nothing in pa_small_q.cm exits early on
// it (the only guard is local_qg >= WG_THREADS), so a thread holding only dummy rows still
// runs the whole online-softmax loop over every KV partition and the epilogue still writes
// its HEAD_SIZE f32 of zeros and its -inf lse. At q_len=6 / GQA-4 / TILE_Q=16 that is 5 of 8
// threads and 10 of 16 partial rows -- ~10 MB of garbage writes at 15 k context, against
// ~32 MB of KV actually read. (Finalization does skip dummy rows, so the cost is write-side.)
//
// Constraints, mirrored by the kernel:
//   * Q_ROWS <= 8, or Q_ROWS a multiple of 8   (ROWS_PER_THREAD is 8 above 8 rows)
//   * Q_ROWS <= 64                             (WG_THREADS <= HEAD_SIZE/REG_K = 8 marshal chunks)
//   * TILE_Q  <= SMALL_Q_THRESHOLD             (no subsequence routed here is longer)
// Overridable downward via OV_GPU_PA_TILE_Q.
inline constexpr int SMALL_Q_MAX_Q_ROWS = 64;

inline int get_small_q_tile_q(int xe_arch, int q_head_chunk_size) {
    const int chunk = std::max(1, q_head_chunk_size);
    int tile_q = std::min<int>(SMALL_Q_THRESHOLD, std::max(1, SMALL_Q_MAX_Q_ROWS / chunk));
    if (const char* env = std::getenv("OV_GPU_PA_TILE_Q")) {
        const int v = std::atoi(env);
        if (v >= 1) {
            tile_q = std::min(tile_q, v);
        }
    }
    // Step down to a legal Q_ROWS. Terminates: chunk*1 <= 8 whenever chunk <= 8, and for
    // chunk > 8 the loop lands on the first multiple-of-8 product.
    while (tile_q > 1) {
        const int q_rows = chunk * tile_q;
        if (q_rows <= 8 || q_rows % 8 == 0) {
            break;
        }
        --tile_q;
    }
    (void)xe_arch;
    return tile_q;
}

// Threads per pa_small_q workgroup, i.e. lws[0]. Must match WG_THREADS in pa_small_q.cm,
// which derives it from the same TILE_Q and Q_head_chunk_size; a mismatch is silent.
inline int get_small_q_wg_threads(int q_head_chunk_size, int tile_q) {
    const int q_rows = std::max(1, q_head_chunk_size) * std::max(1, tile_q);
    return q_rows > 8 ? q_rows / 8 : 1;
}

// MARSHAL_CHUNKS_C in pa_small_q.cm: HEAD_SIZE / REG_K, with REG_K = SystolicDepth *
// VNNI_WIDTH = 16 on every arch the kernel builds for.
inline int get_small_q_marshal_chunks(int head_size) {
    return std::max(1, head_size / 16);
}

// Extra compiled TILE_Q rungs, beyond rung 0 (= get_small_q_tile_q, the shape's legal
// maximum, which always exists). Each costs one more compiled pa_small_q + finalization pair
// at model load, and buys the host a closer fit to the batch's q_len.
inline constexpr std::array<int, 2> SMALL_Q_EXTRA_RUNG_TILE_Q = {8, 6};
inline constexpr size_t SMALL_Q_RUNGS = SMALL_Q_EXTRA_RUNG_TILE_Q.size() + 1;

// Is this TILE_Q something the kernel will accept for this shape?
inline bool is_small_q_tile_q_legal(int q_head_chunk_size, int tile_q) {
    if (tile_q < 1 || tile_q > static_cast<int>(SMALL_Q_THRESHOLD)) {
        return false;
    }
    const int q_rows = std::max(1, q_head_chunk_size) * tile_q;
    return q_rows <= SMALL_Q_MAX_Q_ROWS && (q_rows <= 8 || (q_rows % 8) == 0);
}

// Preference between legal rungs, lower is better.
//
// The cost is set by how the MARSHAL_CHUNKS_C = 8 marshal chunks divide across
// WG_THREADS = q_rows / 8 -- not by TILE_Q, and not monotonically by thread count. Measured
// main-kernel ms at 15 k context, GQA-4, head 128, partition 640 (so TILE_Q = 2 * threads):
//
//   threads   1      2      3      4      5      6      7      8
//   chunks   8      4      3/3/2  2      2..1   2..1   2,1x6  1
//   ms       0.968  0.590  0.523  0.555  0.713  0.890  0.984  0.820
//
// 3 threads is the global minimum and 8 is fine (one chunk each, the loop collapses), but
// 5/6/7 are much worse -- 7 threads splits 8 chunks 2/1/1/1/1/1/1 and everyone waits on the
// pair. So a *smaller* TILE_Q is not automatically better: q_len=4 prefers TILE_Q 6 over 4,
// and q_len=12 prefers 16 over 12.
//
// These are one machine's numbers. The ordering follows chunk-division balance rather than
// anything device-specific, so it should carry, but re-measure before trusting it on a part
// with a different MARSHAL_CHUNKS_C (i.e. a different head size).
inline int small_q_rung_rank(int wg_threads) {
    switch (wg_threads) {
    case 3: return 0;
    case 4: return 1;
    case 2: return 2;
    case 5: return 3;
    case 8: return 4;
    case 6: return 5;
    case 7: return 6;
    default: return 7;  // 1 thread marshals all 8 chunks alone
    }
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

    // small-q decode path (q_len > 1 spec-decoding subsequences)
    size_t small_q_token_count = 0;  // Total (subseq, q-token) pairs routed to pa_small_q
    size_t small_q_tile_count = 0;   // Number of SG tiles = ceil_div(small_q_token_count, TILE_Q)
    // small_q uses its own, larger partition size
    size_t small_q_num_of_partitions = 0;
    // The runtime KV_PARTITION_SIZE this batch chose. gws[2], the kernel scalar and the
    // partial buffer sizes must all derive from THIS field -- they are computed in three
    // different functions, so a divergence would size the work for one partition and index
    // it with another.
    size_t small_q_partition_size = 0;
    int small_q_rung = 0;            // which compiled TILE_Q rung the batch selected
    int small_q_tile_q = 1;          // TILE_Q value chosen at JIT time (must match kernel)
    size_t small_q_max_kv_len = 0;   // Max kv_len across small-q subsequences (for partition count)

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
    SMALL_Q_PARTITIONOUT = 12,
    SMALL_Q_EXPSUMS = 13,
    SMALL_Q_SELECTED_MAPPING = 14,
#else
    // Small-q decode scratch buffers (q_len > 1 spec-decoding path).
    SMALL_Q_PARTITIONOUT = 11,      // f32 partial outputs indexed by (sel_idx, head, partition, head_size)
    SMALL_Q_EXPSUMS = 12,           // f32 lse per (sel_idx, head, partition)
    SMALL_Q_SELECTED_MAPPING = 13,  // triples (orig_seq_idx, q_start, valid_count)
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
        const auto desc = params.typed_desc<paged_attention>();
        const auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
        if (desc->k_head_size == 256 && xe_arch >= 2) {
            constexpr size_t num_team = 8;
            return num_team * get_q_step(params);
        }
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
        }
        return PA_KV_CACHE_BLOCK_SIZE_XATTN;
    }
};

class PagedAttentionGeneratorSingleTokenFinalization : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorSingleTokenFinalization() : PagedAttentionGeneratorBase("pa_single_token_finalization") {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

// Small-q decode kernel for q_len > 1 subsequences. GWS fans the (subseq, q-token,
// kv-partition, head-chunk) combinations out one per SG; reuses pa_single_token's
// per-partition FA layout but indexes everything by a packed sel_idx.
class PagedAttentionGeneratorSmallQ : public PagedAttentionGeneratorBase {
public:
    // rung selects which compiled TILE_Q this stage carries: 0 = get_small_q_tile_q,
    // 1 = get_small_q_tile_q_alt. It is the rung and not the TILE_Q itself because the stage
    // suffix has to be fixed at construction, where params (and so q_head_chunk_size) are not
    // available yet; get_jit_constants resolves the concrete value. The suffix keeps the two
    // variants on distinct entry points so they cannot collide in the kernel cache.
    explicit PagedAttentionGeneratorSmallQ(int rung = 0) : PagedAttentionGeneratorBase("pa_small_q", "_cm_tq" + std::to_string(rung)), _rung(rung) {}

    // pa_small_q wants a *smaller* register file than the other PA kernels. It fits in ~150
    // registers, so 192 is not about spill: at 256 the file allows only 5 threads per XVE,
    // while at 192 it allows 6 *and* leaves IGC enough slack to batch the consume phase's
    // SLM reads (it emits LLLLDDDD instead of a strict LDLDLD chain, giving 4-way
    // memory-level parallelism on SLM). Measured -8.4 % and -11.3 % against 160 in two
    // paired batches; 256 measured +2.3 % / -1.0 %.
    static constexpr int32_t REGISTER_FILE_SIZE = 192;
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + " -cmc -Qxcm_register_file_size=" +
               std::to_string(REGISTER_FILE_SIZE);
    }

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

    // Upper bound compiled into the kernel, NOT the value used. KV_PARTITION_SIZE is a runtime
    // scalar now (see pa_small_q.cm); this only sizes the three prologue lookaside tables, at
    // 3 * (MAX / KV_STEP) * 4 = 480 bytes (7.5 GRF) unconditionally. Raising it is not free --
    // 2048 would cost 24 GRF on a kernel already near the ceiling -- and 640 is the largest
    // partition that ever measured best, so it is also the largest worth allowing.
    //
    // It is also what feeds get_single_token_q_chunking for this stage, so it must stay in the
    // range where that helper still yields q_head_chunk_size 4 (it drops to 2 above 768, which
    // would double the workgroup count and the K/V traffic). update_rt_params asserts the
    // resulting chunking matches the dispatch chunking.
    static constexpr size_t SMALL_Q_PARTITION_MAX = 640;
    //
    // This was 640 for a while, chosen from a 15 k-context sweep. That was overfitting: the
    // partition size sets the workgroup count (WGs = kv_heads * chunks_per_kv * nparts), so a
    // large partition starves the machine at short context. At past_len 512 a 640-token
    // partition yields *one* partition, i.e. 8 workgroups of 3 threads = 24 threads on a part
    // that holds ~160.
    //
    // Measured, main + reduce ms (GQA-4, head 128, cmpr 2, TILE_Q = q_len):
    //
    //           q_len=6                        q_len=16
    //   past   128    256    384    640      128    256    384    640
    //    512  0.042  0.062  0.092  0.113    0.070  0.093  0.077  0.082
    //   2048  0.115  0.122  0.096  0.137    0.190  0.170  0.171  0.188
    //  15360  0.686  0.597  0.586  0.552    1.275  0.997  0.930  0.888
    //
    // Cost of pinning one value, against the best per case:
    //   128 -> worst +44 %, mean +17 %      384 -> worst +118 %, mean +23 %
    //   256 -> worst +47 %, mean +21 %      640 -> worst +168 %, mean +40 %
    //
    // So no constant is good: 640 is the *worst* of the four, and even the best (128) gives up
    // 44 % at 15 k. The real fix is to compile a couple of partition variants and pick on
    // max_context_len, the same way SMALL_Q_EXTRA_RUNG_TILE_Q does for TILE_Q -- 640 would then
    // be selected only for long contexts, where it is worth 12 % at q_len=16. Until then this
    // tracks single-token so short contexts are not penalised.
    //
    // Regression coverage: test_15k_perf_comparison_ov_exp.py::test_small_q_partition_choice.
    static size_t get_partition_size(const bool /*has_xattention*/ = false) {
        return SMALL_Q_PARTITION_MAX;
    }

private:
    int _rung = 0;
};

class PagedAttentionGeneratorSmallQFinalization : public PagedAttentionGeneratorBase {
public:
    // Same rung indexing as PagedAttentionGeneratorSmallQ, and it must be kept in lockstep
    // with it: pa_small_q writes partials at token_row = tile_idx * TILE_Q + t and this
    // kernel decomposes token_row by TILE_Q, so the two stages are only ever executed as a
    // matching pair.
    explicit PagedAttentionGeneratorSmallQFinalization(int rung = 0)
        : PagedAttentionGeneratorBase("pa_small_q_finalization", "_cm_tq" + std::to_string(rung)),
          _rung(rung) {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

private:
    int _rung = 0;
};

// Runtime KV_PARTITION_SIZE for a batch, from the context length.
//
// The partition sets two opposing things: the workgroup count
// (WGs = kv_heads * chunks_per_kv * ceil(context / partition)) and the fp32 partial traffic
// the finalization reads back (proportional to the same partition count). Parallelism wants
// many partitions, traffic wants few, and the balance moves with context -- so no constant is
// right. A value tuned at 15 k left ONE partition at past_len=512, i.e. 8 workgroups of 3
// threads on a part that holds ~160, and ran 3.3x slower.
//
// Measured best partition (main measured + reduce modelled at ~100 GB/s, GQA-4, head 128,
// cmpr 2, TILE_Q = q_len):
//
//   context    512   1024   2048   4096   8192   15360
//   q_len=6    128    256    384    384    640     640
//   q_len=16   384*   256    512*   640*   640     640      (* within 7 % of this table)
//
// Note q_len=6 prefers a smaller partition than q_len=16 at the same context: it runs 3
// threads per workgroup against 8, so it needs more workgroups to fill the machine. The table
// below follows the q_len=6 column, which costs q_len=16 at most 7 %.
//
// Against the per-case best: this table is +7 % worst / +2 % mean, where the previous fixed
// 256 was +47 % / +21 %.
//
// A table and not a formula on purpose: the thresholds were measured on one device with one
// head size (which sets MARSHAL_CHUNKS_C, and with it the whole thread-count cost curve), and
// a table is auditable and regression-testable. Re-measure with
// test_small_q_partition_choice before trusting it elsewhere.
inline size_t pick_small_q_partition(size_t max_context_len, bool /*has_xattention*/) {
    size_t p;
    if (max_context_len <= 768) {
        p = 128;
    } else if (max_context_len <= 1536) {
        p = 256;
    } else if (max_context_len <= 6144) {
        p = 384;
    } else {
        p = 640;
    }
    return std::min(p, PagedAttentionGeneratorSmallQ::SMALL_Q_PARTITION_MAX);
}

// TILE_Q this stage rung resolves to for `params`, or 0 when the rung does not exist for this
// shape (only rung 1 can be absent). Shared by both small-q stages so their JIT constants
// cannot drift apart.
inline int get_small_q_tile_q_for_rung(const kernel_impl_params& params, int rung) {
    const auto desc = params.typed_desc<paged_attention>();
    const int xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    const auto sq_partition_size = PagedAttentionGeneratorSmallQ::get_partition_size(desc->has_xattention);
    const auto chunk = static_cast<int>(get_single_token_q_chunking(params, *desc, sq_partition_size).q_head_chunk_size);
    const int tile_q_max = get_small_q_tile_q(xe_arch, chunk);
    if (rung == 0) {
        return tile_q_max;
    }
    const size_t idx = static_cast<size_t>(rung) - 1;
    if (rung < 1 || idx >= SMALL_Q_EXTRA_RUNG_TILE_Q.size()) {
        return 0;
    }
    const int tile_q = SMALL_Q_EXTRA_RUNG_TILE_Q[idx];
    // A rung at or above the shape's maximum is redundant with rung 0.
    if (tile_q >= tile_q_max || !is_small_q_tile_q_legal(chunk, tile_q)) {
        return 0;
    }
    return tile_q;
}

// Cheapest compiled rung whose TILE_Q still covers max_q_len, as a rung index. Rung 0 always
// qualifies (its TILE_Q is the shape maximum, and SMALL_Q_THRESHOLD bounds q_len by it), so
// this always returns something valid.
inline int pick_small_q_rung(const kernel_impl_params& params, int max_q_len) {
    const auto desc = params.typed_desc<paged_attention>();
    const auto sq_partition_size = PagedAttentionGeneratorSmallQ::get_partition_size(desc->has_xattention);
    const int chunk = static_cast<int>(get_single_token_q_chunking(params, *desc, sq_partition_size).q_head_chunk_size);
    int best_rung = 0;
    int best_rank = small_q_rung_rank(get_small_q_wg_threads(chunk, get_small_q_tile_q_for_rung(params, 0)));
    for (size_t r = 1; r < SMALL_Q_RUNGS; ++r) {
        const int tile_q = get_small_q_tile_q_for_rung(params, static_cast<int>(r));
        if (tile_q < 1 || tile_q < max_q_len) {
            continue;  // not compiled for this shape, or too small to hold the window
        }
        const int rank = small_q_rung_rank(get_small_q_wg_threads(chunk, tile_q));
        if (rank < best_rank) {
            best_rank = rank;
            best_rung = static_cast<int>(r);
        }
    }
    return best_rung;
}

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

    // XAttention metadata is built at WG-level Q-tile granularity, ignoring the
    // head_size=256 worker subdivision used inside the multi-token kernel.
    static size_t get_wg_seq_len(const kernel_impl_params& params) {
        return PagedAttentionGeneratorMultiToken::_wg_size * PagedAttentionGeneratorMultiToken::get_q_step(params);
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
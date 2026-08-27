// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
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

// A TILE_Q the kernel both accepts and marshals evenly.
//
// Legality is the Q_ROWS rule the kernel #errors on. Balance is the extra one: the marshal
// hands out MARSHAL_CHUNKS_C chunks with a strided `c += MARSHAL_TG` loop, so when
// WG_THREADS does not divide the chunk count the chunks pile up unevenly and every thread
// waits on the slowest at the barrier. TILE_Q=6 at GQA-4 is the concrete trap: WG_THREADS=3
// splits 8 chunks 3/3/2, turning a one-round marshal into three, and marshalling on fewer
// threads was already measured slower (see pa_small_q.cm MARSHAL_TG). It is *correct* --
// the strided loop still covers every chunk -- just slower than the rung below it.
inline bool is_small_q_tile_q_balanced(int q_head_chunk_size, int tile_q, int head_size) {
    if (tile_q < 1) {
        return false;
    }
    const int q_rows = std::max(1, q_head_chunk_size) * tile_q;
    if (q_rows > SMALL_Q_MAX_Q_ROWS || (q_rows > 8 && (q_rows % 8) != 0)) {
        return false;
    }
    const int wg_threads = get_small_q_wg_threads(q_head_chunk_size, tile_q);
    return (get_small_q_marshal_chunks(head_size) % wg_threads) == 0;
}

// The rung below get_small_q_tile_q, or 0 if there is none. Since WG_THREADS must divide the
// (power-of-two) chunk count, the balanced rungs are the halvings of the top one: GQA-4 /
// head 128 gives TILE_Q 16 -> 8 -> 4 -> 2.
inline int get_small_q_tile_q_alt(int xe_arch, int q_head_chunk_size, int head_size) {
    const int tile_q_max = get_small_q_tile_q(xe_arch, q_head_chunk_size);
    for (int tile_q = tile_q_max - 1; tile_q >= 1; --tile_q) {
        if (is_small_q_tile_q_balanced(q_head_chunk_size, tile_q, head_size)) {
            return tile_q;
        }
    }
    return 0;
}

// Pick between the two compiled TILE_Q variants for one batch of small-q q_lens.
//
// The alt (smaller) rung wins only when it costs nothing: it must not add a tile to any
// subsequence -- an extra tile is an extra full marshal of the KV partition, the cost the
// large TILE_Q was chosen for -- and it must strictly reduce padded rows. So this can only
// ever delete dummy work, never trade one kind for another. At GQA-4 / head 128 (rungs 16
// and 8): q_len 6 picks 8, q_len 16 picks 16, q_len 9..15 picks 16 (alt would double the
// tiles), and a mixed batch of 6 and 16 picks 16.
inline int pick_small_q_tile_q(int tile_q_max, int tile_q_alt, const std::vector<int>& q_lens) {
    if (tile_q_alt < 1 || tile_q_alt >= tile_q_max || q_lens.empty()) {
        return tile_q_max;
    }
    size_t tiles_max = 0;
    size_t tiles_alt = 0;
    for (const int q_len : q_lens) {
        if (q_len < 1) {
            continue;
        }
        tiles_max += static_cast<size_t>((q_len + tile_q_max - 1) / tile_q_max);
        tiles_alt += static_cast<size_t>((q_len + tile_q_alt - 1) / tile_q_alt);
    }
    const bool no_extra_tiles = tiles_alt == tiles_max;
    const bool fewer_padded_rows = tiles_alt * static_cast<size_t>(tile_q_alt) < tiles_max * static_cast<size_t>(tile_q_max);
    return (no_extra_tiles && fewer_padded_rows) ? tile_q_alt : tile_q_max;
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

    // small-q wants a *larger* partition than single-token, and the same one for both cache
    // layouts. The partition size sets two things at once: the workgroup count (parallelism for
    // this kernel) and the size of the fp32 partial buffer that this kernel writes and the
    // finalization reads back. Those pull in opposite directions, but very unequally -- the main
    // kernel's preference is shallow while the partials dominate the pair's DRAM traffic, so the
    // optimum sits well above single-token's 128/256.
    //
    // Measured (15 k context, GQA-4, head 128, cmpr=2, main + reduce ms):
    //                    block 16          block 256
    //   partition    q=6      q=16      q=6      q=16
    //     128       0.677    1.256       -        -
    //     256       0.572    1.008     0.627    0.983
    //     512       0.517    0.881     0.525    0.869
    //     640       0.506    0.836     0.515    0.848      <- best for every column
    //     768       0.589    0.854     0.581    0.851
    //
    // 640 is best for both q_len and both layouts, so this stays a constant rather than
    // something keyed on q_len.
    //
    // The ceiling is not performance but get_single_token_q_chunking: it sizes q_head_chunk_size
    // against a full-partition-wide rS tile, which the *single-token* kernel holds but small_q
    // does not (its rS_tile is one online tile wide). Past 768 that model shrinks the chunk from
    // 4 to 2, which doubles q_head_chunks_per_kv_head and so doubles the workgroup count and the
    // K/V read traffic. 1024 measures better than 640 in the sandbox only because that harness
    // forces the chunk to 4. Raising this further needs a small-q-specific chunking model first.
    static constexpr size_t SMALL_Q_PARTITION_SIZE = 640;
    static size_t get_partition_size(const bool has_xattention = false) {
        if (SMALL_Q_PARTITION_SIZE == 0) {
            return PagedAttentionGeneratorSingleToken::get_partition_size(has_xattention);
        }
        return SMALL_Q_PARTITION_SIZE;
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

// TILE_Q this stage rung resolves to for `params`, or 0 when the rung does not exist for this
// shape (only rung 1 can be absent). Shared by both small-q stages so their JIT constants
// cannot drift apart.
inline int get_small_q_tile_q_for_rung(const kernel_impl_params& params, int rung) {
    const auto desc = params.typed_desc<paged_attention>();
    const int xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    const auto sq_partition_size = PagedAttentionGeneratorSmallQ::get_partition_size(desc->has_xattention);
    const auto chunk = static_cast<int>(get_single_token_q_chunking(params, *desc, sq_partition_size).q_head_chunk_size);
    const int head_size = static_cast<int>(desc->k_head_size);
    return rung == 0 ? get_small_q_tile_q(xe_arch, chunk) : get_small_q_tile_q_alt(xe_arch, chunk, head_size);
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
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../primitive_cm_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/qsa.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "primitive_inst.h"
#include "qsa_stage_events.hpp"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::cm {

using namespace cldnn;
using QSAIn = ov::intel_gpu::op::QSA::InputIdx;

namespace {

constexpr size_t kv_update_wg_size = 16;
constexpr size_t prepare_wg_size = 16;
constexpr size_t partition_blocks = 512;  // summaries scanned per decode Q2 partition
constexpr size_t score_wg_size = 16;      // threads per query tile in the prefill dpas Q2
constexpr size_t dpas_query_tile = 16;    // dpas N axis == GRF width on Xe2
constexpr size_t dpas_block_tile = 8;     // dpas RepeatCount (M axis)
constexpr size_t dpas_k_step = 16;          // fp16 K elements per dpas
constexpr size_t dpas_wide_cols = 32;       // halfs per 2D block row; the hardware caps a row at 64 B
constexpr size_t max_slm_bytes = 64 * 1024;
constexpr size_t topk_wg_size = 16;          // qsa_topk_radix_wg cooperative WG size
// This is the kernel's own upper bound: compiled once with the maximum, it degrades to
// an uncached global scan only above 2048 SIMD64-aligned keys/worker, never producing a
// wrong selection at any context length.
constexpr size_t topk_cache_capacity = 2048;
constexpr size_t split_partitions = 16;      // decode split-selected-KV Q3 partitions
// Only route decode Q3 through the split path at/above this past length; shorter decode
// contexts are already cheap on the scalar kernel and do not need the extra reduce pass.
constexpr size_t split_decode_min_past = 2048;

/// See qsa_stage_events.hpp: a debug-only record of the per-kernel events of the *current*
/// QSA::execute() call, cleared at the start of every call. thread_local because network
/// execution can happen from different threads across tests/streams; TU-local (anonymous
/// namespace) storage, exposed to other translation units only via the accessor function
/// defined at the end of this file.
thread_local std::vector<std::pair<std::string, event::ptr>> g_qsa_stage_events;

void record_stage_event(const char* name, const event::ptr& ev) {
    if (ev != nullptr) {
        g_qsa_stage_events.emplace_back(name, ev);
    }
}

/// Row stride of the [num_tokens, max_blocks] score buffer. The systolic scorer writes an
/// entire 8-block dpas tile at once, so the last tile of a row can reach up to
/// `dpas_block_tile - 1` entries past the block count; without this padding that store
/// lands in the next token's row and silently corrupts its top-k. Every kernel takes the
/// stride as a scalar, so padding it keeps all of them consistent, and the padded entries
/// are never read because the selection is bounded by each token's own n_complete.
size_t block_scores_stride(size_t max_blocks) {
    return (max_blocks + dpas_block_tile - 1) / dpas_block_tile * dpas_block_tile;
}

/// Whether the systolic prefill scoring kernel can be used for this configuration.
/// Everything it needs is a compile-time property of the descriptor plus the GRF width,
/// so the decision is stable across iterations; when it does not hold the scalar
/// qsa_score_topk_fused kernel is used instead (QSA never falls back to PA/SDPA).
bool can_use_dpas_score(const qsa& desc, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;  // REG_N is 8 on Xe1, the kernel assumes a 16-wide N axis
    }
    if (desc.indexer_head_dim % dpas_k_step != 0) {
        return false;
    }
    if (desc.indexer_head_dim * sizeof(ov::float16) < 64) {
        return false;  // 2D block load surface width minimum
    }
    const auto summaries_per_page = desc.pa_block_size / desc.compress_ratio;
    if (summaries_per_page == 0 || (summaries_per_page % dpas_block_tile != 0 && dpas_block_tile % summaries_per_page != 0)) {
        return false;  // the A tile must be whole pages, or a whole tile must fit in a page
    }
    const auto staged_bytes = desc.indexer_heads * (desc.indexer_head_dim / dpas_k_step) * dpas_k_step * dpas_query_tile * sizeof(ov::float16);
    return staged_bytes <= max_slm_bytes;
}

// Upper bound on the BLOCK_SCORES scratch buffer's byte size, regardless of how many
// query rows a call actually has, so a 128K-token prefill never allocates the tens-of-GB
// [tokens, max_blocks] f32 buffer an unbounded design would need. Q2 instead runs as a
// sequence of query-row chunks, each scoring/selecting into the SAME bounded buffer
// before the next chunk reuses it.
constexpr size_t score_budget_bytes = 128 * 1024 * 1024;

/// Query-row capacity of the (possibly bounded) BLOCK_SCORES buffer for this shape/device.
/// Only bounded when the row-chunk-aware kernels (qsa_score_tile_dpas for prefill,
/// qsa_score_partition + qsa_topk_radix_wg for decode) will actually be used, i.e. when
/// can_use_dpas_score() holds; the legacy qsa_score_topk_fused fallback (Xe1, or an
/// unsupported dpas shape) is not row-chunk aware and still needs the full [tokens] rows.
/// Uses ONLY static input layouts (never runtime past_lens/subsequence_begins) so that
/// get_internal_buffer_descs() (allocation time) and update_rt_params() (chunk planning,
/// every call) always agree on the exact same capacity.
size_t score_scratch_rows(const kernel_impl_params& params, const qsa& desc) {
    const auto tokens = std::max<size_t>(1, params.get_input_layout(QSAIn::QUERY).get_shape()[0]);
    if (!can_use_dpas_score(desc, params.get_device_info())) {
        return tokens;
    }
    const auto& kcache = params.get_input_layout(QSAIn::KEY_CACHE).get_shape();
    const auto max_blocks = block_scores_stride(std::max<size_t>(1, kcache[0] * desc.pa_block_size / desc.compress_ratio));
    const size_t row_bytes = max_blocks * 4;
    const size_t budget_rows = std::max<size_t>(1, score_budget_bytes / row_bytes);
    return std::min(tokens, budget_rows);
}

/// Heads that share one dpas tile in the systolic sparse attention.
///
/// The dpas N axis must be filled with columns that share the A operand (the KV tile).
/// Filling it with 16 query tokens makes the tile attend over the union of 16 independent
/// per-token top-k selections - measured ~15x the useful work, which no amount of inner
/// loop tuning recovers. Heads of one token instead share that token's selection, and
/// under GQA the `n_rep = Hq / Hkv` heads of a group also share the same K/V, so a tile of
/// HT heads x (16 / HT) tokens both shrinks the union to 16 / HT selections and reads K/V
/// once for HT heads where the scalar kernel re-reads them per head. HT must divide 16 and
/// divide n_rep, so that a head group never straddles a KV head.
size_t dpas_heads_per_tile(const qsa& desc) {
    const auto n_rep = desc.heads_num / desc.kv_heads_num;
    for (size_t ht : {size_t{16}, size_t{8}, size_t{4}, size_t{2}}) {
        if (n_rep % ht == 0) {
            return ht;
        }
    }
    return 1;
}

/// Threads that cooperate on one (query tile, head) in the systolic sparse attention.
///
/// Two constraints pick W. The register file is the first: the output accumulator alone is
/// `dpas_query_tile * Dv * 4` bytes - 16 KB at Dv = 256, i.e. the whole 256-GRF file - and
/// the kernel also stages the step's K and V tiles whole so that each union block arrives
/// in as few wide messages as possible. Splitting the head dimension across W threads
/// divides all of those by W, at the cost of one SLM reduction of the QK tile per step
/// (cm_pa_xe2.hpp does the same for head_size == 256) - and that reduction costs O(W) SLM
/// traffic per worker, so the SMALLEST W that fits is the right choice.
/// Returns 0 when no supported split fits, which disables the systolic path.
size_t dpas_attention_workers(const qsa& desc) {
    // Deliberately below the 16 KB file: the QK tile, the transposed P and the softmax
    // state still have to fit alongside rQ, rO and the staged K/V.
    constexpr size_t reg_budget = 12 * 1024;
    for (size_t w : {size_t{1}, size_t{2}, size_t{4}, size_t{8}}) {
        if (desc.k_head_size % (w * dpas_k_step) != 0 || desc.v_head_size % (w * dpas_query_tile) != 0) {
            continue;
        }
        const auto dh_w = desc.k_head_size / w;
        const auto dv_w = desc.v_head_size / w;
        if (dh_w % dpas_wide_cols != 0 || dv_w % dpas_wide_cols != 0) {
            continue;  // the slice must be a whole number of 64-byte block rows
        }
        const auto rq = dh_w * dpas_query_tile * sizeof(ov::float16);
        const auto ro = dpas_query_tile * dv_w * sizeof(float);
        const auto kv = dpas_k_step * (dh_w + dv_w) * sizeof(ov::float16);  // staged K and V
        if (rq + ro + kv <= reg_budget) {
            return w;
        }
    }
    return 0;
}

/// Whether the systolic prefill sparse-attention kernel can be used.
/// It needs a 16-wide dpas N axis, head sizes aligned to the dpas tiles, and an even
/// compress ratio (a QSA block must cover whole VNNI row pairs of the V operand).
bool can_use_dpas_attention(const qsa& desc, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;
    }
    if (desc.k_head_size % dpas_k_step != 0 || desc.v_head_size % dpas_query_tile != 0) {
        return false;
    }
    if (desc.k_head_size * sizeof(ov::float16) < 64 || desc.v_head_size * sizeof(ov::float16) < 64) {
        return false;  // 2D block load surface width minimum
    }
    if (desc.compress_ratio % 2 != 0 || dpas_k_step % desc.compress_ratio != 0) {
        return false;  // four QSA blocks must pack exactly one 16-token dpas K step
    }
    return dpas_attention_workers(desc) != 0;
}

/// Whether the systolic Q1 indexer projection kernel (qsa_project_dpas) can be used.
/// It needs a 16-wide dpas N axis and a hidden size / projection width aligned to its
/// M8/K32 tiles; qsa_prepare falls back to its own scalar-tiled projection otherwise.
bool can_use_dpas_projection(const qsa& desc, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;  // the kernel asserts a 512-bit (16-wide) GRF
    }
    if (desc.hidden_size % 32 != 0) {
        return false;  // 2D surfaces need >=64B width / 64B-aligned row pitch
    }
    const auto proj_width = (desc.indexer_heads + 1) * desc.indexer_head_dim;
    return proj_width % dpas_block_tile == 0;  // J must be divisible by the M8 output tile
}

/// Whether the GQA-aware systolic Q3 (qsa_sparse_attention_gqa_dpas) can be used.
/// Needs an integral GQA group that fits the N16 dpas tile, W4 head-dimension slices
/// aligned to 32 halfs, head sizes within the verified <=256 range, and a compress ratio
/// the kernel can pack into one 16-token dpas K step. When unsupported, QSACmImpl falls
/// back to the older union-based qsa_sparse_attention_dpas, then to the scalar kernel.
bool can_use_gqa_dpas_attention(const qsa& desc, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;
    }
    if (desc.kv_heads_num == 0 || desc.heads_num % desc.kv_heads_num != 0) {
        return false;
    }
    const auto n_rep = desc.heads_num / desc.kv_heads_num;
    if (n_rep < 1 || n_rep > 16) {
        return false;
    }
    constexpr size_t gqa_workers = 4;  // GQA_W in qsa_sparse_attention_gqa_dpas.cm
    if (desc.k_head_size % (gqa_workers * 32) != 0 || desc.v_head_size % (gqa_workers * 32) != 0) {
        return false;
    }
    if (std::max(desc.k_head_size, desc.v_head_size) > 256) {
        return false;  // above the verified accumulator range
    }
    if (desc.compress_ratio != 2 && desc.compress_ratio != 4 && desc.compress_ratio != 8 && desc.compress_ratio != 16) {
        return false;
    }
    if (desc.pa_block_size == 0 || desc.pa_block_size % desc.compress_ratio != 0) {
        return false;
    }
    return desc.is_causal;
}

constexpr auto get_qsa_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
}

std::vector<int32_t> read_i32_input(const primitive_inst& instance, size_t input_idx) {
    auto mem = instance.input_memory_ptr(input_idx);
    mem_lock<int32_t, mem_lock_type::read> lock(mem, instance.get_network().get_stream());
    return {lock.begin(), lock.end()};
}

QSARuntimeParams* rt_of(ImplRuntimeParams* rt) {
    auto* p = static_cast<QSARuntimeParams*>(rt);
    OPENVINO_ASSERT(p != nullptr, "[GPU][CM] qsa runtime params are not initialized");
    return p;
}

/// JIT constants shared by every QSA kernel family.
JitConstants make_qsa_jit(const std::shared_ptr<const qsa>& desc) {
    JitConstants jit;
    jit.make("QSA_HEADS", desc->heads_num);
    jit.make("QSA_KV_HEADS", desc->kv_heads_num);
    jit.make("QSA_K_HEAD_SIZE", desc->k_head_size);
    jit.make("QSA_V_HEAD_SIZE", desc->v_head_size);
    jit.make("QSA_HIDDEN", desc->hidden_size);
    jit.make("QSA_IDX_HEADS", desc->indexer_heads);
    jit.make("QSA_IDX_HEAD_DIM", desc->indexer_head_dim);
    jit.make("QSA_IDX_ROTARY_DIM", desc->indexer_rotary_dim);
    jit.make("QSA_COMPRESS_RATIO", desc->compress_ratio);
    jit.make("QSA_BLOCK_TOPK", desc->block_topk);
    jit.make("QSA_BUDGET", desc->budget);
    jit.make("QSA_MAX_SELECTED", desc->max_selected_tokens());
    jit.make("QSA_PA_BLOCK_SIZE", desc->pa_block_size);
    jit.make("QSA_HAS_OUTPUT_GATE", desc->has_output_gate ? 1 : 0);
    jit.make("QSA_IS_CAUSAL", desc->is_causal ? 1 : 0);
    jit.add(make_jit_constant("QSA_SCALE", desc->scale));
    jit.add(make_jit_constant("QSA_IDX_SCALE", 1.0f / std::sqrt(static_cast<float>(desc->indexer_head_dim))));
    jit.add(make_jit_constant("QSA_IDX_NORM_EPS", desc->indexer_norm_eps));
    // Order-preserving dense-prefix bypass (validated correctness-safe optimization): rows
    // whose complete-block count does not exceed block_topk select every block via arange,
    // without the finalizer ever reading (or, with QSA_DENSE_SCORE_BYPASS, the score kernel
    // ever writing) scores for that row. See qsa_common.hpp for the paired-flag guard.
    jit.make("QSA_DENSE_BYPASS", 1);
    jit.make("QSA_DENSE_SCORE_BYPASS", 1);
    return jit;
}

void push_u32(KernelData& kd, size_t value) {
    scalar_desc s;
    s.t = scalar_desc::Types::UINT32;
    s.v.u32 = static_cast<uint32_t>(value);
    kd.params.scalars.push_back(s);
}

/// Number of elements between two consecutive tokens of a token-major flat buffer.
size_t token_pitch(const layout& l) {
    const auto& shape = l.get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "[GPU][CM] qsa expects rank-2 token-major inputs");
    return shape[1];
}

// ===========================================================================
// Q0 - main KV cache update
// ===========================================================================
class QSAKVCacheUpdateGenerator : public KernelGenerator {
public:
    QSAKVCacheUpdateGenerator() : KernelGenerator("qsa_kv_cache_update", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_qsa_jit(params.typed_desc<qsa>()));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::KEY});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::VALUE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::VALUE_CACHE});
        for (uint32_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<qsa>();
            auto* rtp = rt_of(rt);

            const size_t wg_count = (rtp->num_tokens + kv_update_wg_size - 1) / kv_update_wg_size;
            auto& wgs = kd.params.workGroups;
            wgs.global = {1, desc->kv_heads_num, std::max<size_t>(1, wg_count) * kv_update_wg_size};
            wgs.local = {1, 1, kv_update_wg_size};

            kd.params.scalars.clear();
            push_u32(kd, token_pitch(params.get_input_layout(QSAIn::KEY)));
            push_u32(kd, params.get_input_layout(QSAIn::KEY).get_linear_offset());
            push_u32(kd, token_pitch(params.get_input_layout(QSAIn::VALUE)));
            push_u32(kd, params.get_input_layout(QSAIn::VALUE).get_linear_offset());
            push_u32(kd, rtp->num_seqs);
        }};
    }
};

// ===========================================================================
// Q1 - indexer prepare
// ===========================================================================
class QSAPrepareGenerator : public KernelGenerator {
public:
    QSAPrepareGenerator() : KernelGenerator("qsa_prepare", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_qsa_jit(params.typed_desc<qsa>()));
        jit.make("QSA_PREPARE_WG", prepare_wg_size);
        if (can_use_dpas_projection(*params.typed_desc<qsa>(), params.get_device_info())) {
            jit.make("QSA_PREPROJECTED", 1);
        }
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (can_use_dpas_projection(*params.typed_desc<qsa>(), params.get_device_info())) {
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::PROJECTED});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::MIXED});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_QK_WEIGHT});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_Q_NORM_WEIGHT});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_K_NORM_WEIGHT});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::ROTARY_COS_SIN});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::POSITION_IDS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_RAW_K_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::PREPARE_MAP});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            auto* rtp = rt_of(rt);
            const size_t wg_count = std::max<size_t>(1, rtp->prepare_wg_count);

            auto& wgs = kd.params.workGroups;
            wgs.global = {1, 1, wg_count * prepare_wg_size};
            wgs.local = {1, 1, prepare_wg_size};

            kd.params.scalars.clear();
            push_u32(kd, rtp->prepare_wg_count);
        }};
    }
};

// ===========================================================================
// Q1 systolic variant - indexer Q/K projection on cm_dpas
// ===========================================================================
// Feeds QSAPrepareGenerator's QSA_PREPROJECTED path: [tokens,D] x [J,D]^T -> [tokens,J]
// FP32, so the fused qsa_prepare kernel only has to do norm/RoPE/pooling/cache-layout
// work. Only dispatched when can_use_dpas_projection() holds; qsa_prepare's own
// scalar-tiled projection remains the fallback (unchanged) for older architectures.
class QSAProjectDpasGenerator : public KernelGenerator {
public:
    QSAProjectDpasGenerator() : KernelGenerator("qsa_project_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_qsa_jit(params.typed_desc<qsa>()));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::MIXED});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_QK_WEIGHT});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::PROJECTED});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<qsa>();
            auto* rtp = rt_of(rt);
            const size_t proj_width = (desc->indexer_heads + 1) * desc->indexer_head_dim;

            auto& wgs = kd.params.workGroups;
            // gws = [J/8, ceil(tokens/16), 1]; one work-item owns an M8 x N16 dpas tile.
            const size_t tokens = std::max<size_t>(1, rtp->num_tokens);
            wgs.global = {proj_width / dpas_block_tile, (tokens + dpas_query_tile - 1) / dpas_query_tile, 1};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
        }};
    }
};

// ===========================================================================
// Q2 prefill - fused score + top-k
// ===========================================================================
class QSAScoreTopkFusedGenerator : public KernelGenerator {
public:
    QSAScoreTopkFusedGenerator() : KernelGenerator("qsa_score_topk_fused", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_qsa_jit(params.typed_desc<qsa>()));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::BLOCK_SCORES});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_BLOCKS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_COUNTS});
        for (uint32_t i = 0; i < 3; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            auto* rtp = rt_of(rt);
            auto& wgs = kd.params.workGroups;
            wgs.global = {std::max<size_t>(1, rtp->num_tokens), 1, 1};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, rtp->max_blocks);
        }};
    }
};

// ===========================================================================
// Q2 prefill, systolic variant - fused score + top-k on cm_dpas
// ===========================================================================
class QSAScoreTileDpasGenerator : public KernelGenerator {
public:
    QSAScoreTileDpasGenerator() : KernelGenerator("qsa_score_tile_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_qsa_jit(params.typed_desc<qsa>()));
        jit.make("QSA_SCORE_WG", score_wg_size);
        // Always row-chunk aware: QSACmImpl::execute() always dispatches this kernel in a
        // per-chunk loop (a shape small enough to fit the score budget in one chunk just
        // gets a single trivial chunk covering the whole row range).
        jit.make("QSA_SCORE_ROW_CHUNK", 1);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SCORE_TILE_MAP});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::BLOCK_SCORES});
        // (num_tiles, max_blocks, row_begin, row_end, tile_offset), matching the
        // QSA_SCORE_ROW_CHUNK kernel signature.
        for (uint32_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            auto* rtp = rt_of(rt);
            auto& wgs = kd.params.workGroups;
            // One work-group owns one 16-query tile; its threads split the block range.
            // Only the current chunk's tile slice is dispatched.
            wgs.global = {std::max<size_t>(1, rtp->chunk_tile_count) * score_wg_size, 1, 1};
            wgs.local = {score_wg_size, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->chunk_tile_count);
            push_u32(kd, rtp->max_blocks);
            push_u32(kd, rtp->chunk_row_begin);
            push_u32(kd, rtp->chunk_row_end);
            push_u32(kd, rtp->chunk_tile_offset);
        }};
    }
};

// ===========================================================================
// Q2 decode pass 1 - score partition
// ===========================================================================
class QSAScorePartitionGenerator : public KernelGenerator {
public:
    QSAScorePartitionGenerator() : KernelGenerator("qsa_score_partition", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_qsa_jit(params.typed_desc<qsa>()));
        jit.make("QSA_PARTITION_BLOCKS", partition_blocks);
        // Always row-chunk aware; see QSAScoreTileDpasGenerator for the rationale.
        jit.make("QSA_SCORE_ROW_CHUNK", 1);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::BLOCK_SCORES});
        // (num_tokens, num_seqs, max_blocks, row_begin, row_end), matching the
        // QSA_SCORE_ROW_CHUNK kernel signature.
        for (uint32_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            auto* rtp = rt_of(rt);
            auto& wgs = kd.params.workGroups;
            // Partitions run along the block range only. Splitting along Hidx is forbidden
            // because ReLU precedes the head reduction. Dim0 is only the current chunk's
            // row range; the kernel adds chunk_row_begin back to recover the flat token id.
            const size_t chunk_rows = std::max<size_t>(1, rtp->chunk_row_end - rtp->chunk_row_begin);
            wgs.global = {chunk_rows, 1, std::max<size_t>(1, rtp->num_partitions)};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, rtp->max_blocks);
            push_u32(kd, rtp->chunk_row_begin);
            push_u32(kd, rtp->chunk_row_end);
        }};
    }
};

// ===========================================================================
// Q2 decode pass 2, cooperative variant - fast-wg16-cached exact top-k
// ===========================================================================
// One work-group of QSA_TOPK_WG threads cooperatively scans each row instead of a
// single thread, with an opt-in per-worker key cache. QSA_TOPK_CACHE_CAPACITY is compiled
// at its own maximum (2048): the kernel falls back to an uncached global scan above that
// many SIMD64-aligned keys/worker, so this is always safe and never an approximation, for
// any context length. Used for both the decode and the prefill-dpas selection paths.
class QSATopkRadixWgGenerator : public KernelGenerator {
public:
    QSATopkRadixWgGenerator() : KernelGenerator("qsa_topk_radix_wg", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_qsa_jit(params.typed_desc<qsa>()));
        jit.make("QSA_TOPK_WG", topk_wg_size);
        jit.make("QSA_TOPK_CACHE_CAPACITY", topk_cache_capacity);
        // Always row-chunk aware; see QSAScoreTileDpasGenerator for the rationale.
        jit.make("QSA_SCORE_ROW_CHUNK", 1);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::BLOCK_SCORES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_BLOCKS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_COUNTS});
        // (num_tokens, num_seqs, max_blocks, row_begin, row_end), matching the
        // QSA_SCORE_ROW_CHUNK kernel signature.
        for (uint32_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            auto* rtp = rt_of(rt);
            auto& wgs = kd.params.workGroups;
            // One work-group per token in the current chunk; its QSA_TOPK_WG threads
            // cooperatively scan the row.
            const size_t chunk_rows = std::max<size_t>(1, rtp->chunk_row_end - rtp->chunk_row_begin);
            wgs.global = {chunk_rows * topk_wg_size, 1, 1};
            wgs.local = {topk_wg_size, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, rtp->max_blocks);
            push_u32(kd, rtp->chunk_row_begin);
            push_u32(kd, rtp->chunk_row_end);
        }};
    }
};

// ===========================================================================
// Q3 prefill, systolic variant - query-tile sparse attention on cm_dpas
// ===========================================================================
class QSASparseAttentionDpasGenerator : public KernelGenerator {
public:
    QSASparseAttentionDpasGenerator() : KernelGenerator("qsa_sparse_attention_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<qsa>();
        jit.add(make_qsa_jit(desc));
        jit.make("QSA_ATT_WORKERS", std::max<size_t>(1, dpas_attention_workers(*desc)));
        jit.make("QSA_HEADS_PER_TILE", dpas_heads_per_tile(*desc));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::VALUE_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_BLOCKS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_COUNTS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SCORE_TILE_MAP});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        for (uint32_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<qsa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            // One work-group is one (head group, TT-token sub-tile): its threads split the
            // head dimension and exchange QK partial sums, so they must not straddle tiles.
            // Each 16-token entry of the shared tile map covers HT sub-tiles of 16/HT
            // tokens, so dimension 1 enumerates (map entry, sub-tile) pairs.
            const size_t workers = std::max<size_t>(1, dpas_attention_workers(*desc));
            const size_t heads_per_tile = dpas_heads_per_tile(*desc);
            wgs.global = {(desc->heads_num / heads_per_tile) * workers,
                          std::max<size_t>(1, rtp->score_tile_count) * heads_per_tile,
                          1};
            wgs.local = {workers, 1, 1};

            const auto& q_layout = params.get_input_layout(QSAIn::QUERY);
            const size_t q_head_stride = desc->has_output_gate ? 2 * desc->k_head_size : desc->k_head_size;
            const size_t gate_delta = desc->has_output_gate ? desc->k_head_size : 0;

            kd.params.scalars.clear();
            push_u32(kd, rtp->score_tile_count);
            push_u32(kd, q_layout.get_linear_offset());
            push_u32(kd, token_pitch(q_layout));
            push_u32(kd, q_head_stride);
            push_u32(kd, gate_delta);
        }};
    }
};

// ===========================================================================
// Q3 - sparse attention
// ===========================================================================
class QSASparseAttentionGenerator : public KernelGenerator {
public:
    QSASparseAttentionGenerator() : KernelGenerator("qsa_sparse_attention", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_qsa_jit(params.typed_desc<qsa>()));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::VALUE_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_BLOCKS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_COUNTS});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        for (uint32_t i = 0; i < 6; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<qsa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->heads_num, std::max<size_t>(1, rtp->num_tokens), 1};
            wgs.local = {1, 1, 1};

            // Q pitch/offset are runtime scalars so that a fused q_proj output of
            // [query(Dh) | gate(Dh)] per head is consumed without any repacking.
            const auto& q_layout = params.get_input_layout(QSAIn::QUERY);
            const size_t q_pitch = token_pitch(q_layout);
            const size_t q_head_stride = desc->has_output_gate ? 2 * desc->k_head_size : desc->k_head_size;
            const size_t gate_delta = desc->has_output_gate ? desc->k_head_size : 0;

            kd.params.scalars.clear();
            // Order must match the kernel signature:
            // (num_tokens, num_seqs, q_base, q_pitch, q_head_stride, gate_delta)
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, q_layout.get_linear_offset());
            push_u32(kd, q_pitch);
            push_u32(kd, q_head_stride);
            push_u32(kd, gate_delta);
        }};
    }
};

// ===========================================================================
// Q3 prefill, GQA-aware systolic variant - one token x one KV head per WG on cm_dpas
// ===========================================================================
// Preferred over QSASparseAttentionDpasGenerator whenever can_use_gqa_dpas_attention()
// holds: the dpas N axis is filled with the GQA replication group (all n_rep query heads
// of one token share one KV read), not a union of several tokens' independent top-k
// selections. QSA_GQA_HEAD_MAJOR/QSA_GQA_VECTOR_META enable KV-head-major work-group
// launch order plus batched masked metadata gathers, which improve KV cache locality on
// long contexts. Neither changes the score/softmax arithmetic.
class QSASparseAttentionGqaDpasGenerator : public KernelGenerator {
public:
    QSASparseAttentionGqaDpasGenerator() : KernelGenerator("qsa_sparse_attention_gqa_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_qsa_jit(params.typed_desc<qsa>()));
        jit.make("QSA_GQA_HEAD_MAJOR", 1);
        jit.make("QSA_GQA_VECTOR_META", 1);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::VALUE_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_BLOCKS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_COUNTS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::TOKEN_MAP});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        for (uint32_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<qsa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            // QSA_GQA_HEAD_MAJOR=1: dim0 enumerates (token, W4 worker), dim1 the KV head,
            // so consecutive work-groups share a KV head.
            const size_t tokens = std::max<size_t>(1, rtp->num_tokens);
            wgs.global = {tokens * 4, desc->kv_heads_num, 1};
            wgs.local = {4, 1, 1};

            const auto& q_layout = params.get_input_layout(QSAIn::QUERY);
            const size_t q_head_stride = desc->has_output_gate ? 2 * desc->k_head_size : desc->k_head_size;
            const size_t gate_delta = desc->has_output_gate ? desc->k_head_size : 0;

            kd.params.scalars.clear();
            // Order must match the kernel signature:
            // (num_entries, q_base, q_pitch, q_head_stride, gate_delta)
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, q_layout.get_linear_offset());
            push_u32(kd, token_pitch(q_layout));
            push_u32(kd, q_head_stride);
            push_u32(kd, gate_delta);
        }};
    }
};

// ===========================================================================
// Q3 decode, split-selected-KV variant, pass 1 - per-partition partial softmax
// ===========================================================================
// Opt-in for long decode (see split_decode_min_past): partitions each token's selected
// KV across QSA_SPLIT_PARTITIONS independent producers so batch=1 decode is not bound by
// one work-item per (head, token). qsa_sparse_attention_split_reduce.cm merges the
// partial (max, sum, weighted-V) state exactly; neither kernel changes the causal/softmax
// semantics of the scalar QSASparseAttentionGenerator it complements.
class QSASparseAttentionSplitGenerator : public KernelGenerator {
public:
    QSASparseAttentionSplitGenerator() : KernelGenerator("qsa_sparse_attention_split", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_qsa_jit(params.typed_desc<qsa>()));
        jit.make("QSA_SPLIT_PARTITIONS", split_partitions);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::VALUE_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_BLOCKS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SELECTED_COUNTS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SPLIT_ACC});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SPLIT_ML});
        for (uint32_t i = 0; i < 6; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<qsa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->heads_num, std::max<size_t>(1, rtp->num_tokens), split_partitions};
            wgs.local = {1, 1, 1};

            const auto& q_layout = params.get_input_layout(QSAIn::QUERY);
            const size_t q_head_stride = desc->has_output_gate ? 2 * desc->k_head_size : desc->k_head_size;
            const size_t gate_delta = desc->has_output_gate ? desc->k_head_size : 0;

            kd.params.scalars.clear();
            // Order must match the kernel signature:
            // (num_tokens, num_seqs, q_base, q_pitch, q_head_stride, gate_delta)
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, q_layout.get_linear_offset());
            push_u32(kd, token_pitch(q_layout));
            push_u32(kd, q_head_stride);
            push_u32(kd, gate_delta);
        }};
    }
};

// ===========================================================================
// Q3 decode, split-selected-KV variant, pass 2 - exact online-softmax reduce
// ===========================================================================
class QSASparseAttentionSplitReduceGenerator : public KernelGenerator {
public:
    QSASparseAttentionSplitReduceGenerator() : KernelGenerator("qsa_sparse_attention_split_reduce", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_qsa_jit(params.typed_desc<qsa>()));
        jit.make("QSA_SPLIT_PARTITIONS", split_partitions);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SPLIT_ACC});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::SPLIT_ML});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        for (uint32_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<qsa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->heads_num, std::max<size_t>(1, rtp->num_tokens), 1};
            wgs.local = {1, 1, 1};

            const auto& q_layout = params.get_input_layout(QSAIn::QUERY);
            const size_t q_head_stride = desc->has_output_gate ? 2 * desc->k_head_size : desc->k_head_size;
            const size_t gate_delta = desc->has_output_gate ? desc->k_head_size : 0;

            kd.params.scalars.clear();
            // Order must match the kernel signature:
            // (num_tokens, q_base, q_pitch, q_head_stride, gate_delta)
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, q_layout.get_linear_offset());
            push_u32(kd, token_pitch(q_layout));
            push_u32(kd, q_head_stride);
            push_u32(kd, gate_delta);
        }};
    }
};

// ---------------------------------------------------------------------------

class QSACmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::QSACmImpl)

    Stage::Ptr kv_cache_update = make_stage<QSAKVCacheUpdateGenerator>();
    Stage::Ptr project_dpas = make_stage<QSAProjectDpasGenerator>();
    Stage::Ptr prepare = make_stage<QSAPrepareGenerator>();
    Stage::Ptr score_topk_fused = make_stage<QSAScoreTopkFusedGenerator>();
    Stage::Ptr score_tile_dpas = make_stage<QSAScoreTileDpasGenerator>();
    Stage::Ptr score_partition = make_stage<QSAScorePartitionGenerator>();
    Stage::Ptr topk_radix_wg = make_stage<QSATopkRadixWgGenerator>();
    Stage::Ptr sparse_attention = make_stage<QSASparseAttentionGenerator>();
    Stage::Ptr sparse_attention_dpas = make_stage<QSASparseAttentionDpasGenerator>();
    Stage::Ptr sparse_attention_gqa_dpas = make_stage<QSASparseAttentionGqaDpasGenerator>();
    Stage::Ptr sparse_attention_split = make_stage<QSASparseAttentionSplitGenerator>();
    Stage::Ptr sparse_attention_split_reduce = make_stage<QSASparseAttentionSplitReduceGenerator>();

    QSACmImpl() : PrimitiveImplCM(QSACMImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<QSARuntimeParams>();
    }

    // PrimitiveImplOCL::make_deep_copy() intentionally drops m_rt_params, so the
    // accessor has to be able to recreate it lazily.
    QSARuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<QSARuntimeParams>();
        }
        return *static_cast<QSARuntimeParams*>(m_rt_params.get());
    }

    explicit QSACmImpl(const RuntimeParams& params) : QSACmImpl() {
        const bool dpas_score = use_dpas_score(params);
        const bool dpas_attention = use_dpas_attention(params);
        const bool dpas_projection = use_dpas_projection(params);
        const bool gqa_dpas_attention = use_gqa_dpas_attention(params);
        add_stage(kv_cache_update, params);
        if (dpas_projection) {
            add_stage(project_dpas, params);
        }
        add_stage(prepare, params);
        add_stage(score_topk_fused, params);
        if (dpas_score) {
            add_stage(score_tile_dpas, params);
        }
        add_stage(score_partition, params);
        // No capability gate: qsa_topk_radix_wg has no cm_dpas dependency, so it always
        // replaces qsa_topk_finalization for both the decode and prefill-dpas selection.
        add_stage(topk_radix_wg, params);
        add_stage(sparse_attention, params);
        if (gqa_dpas_attention) {
            add_stage(sparse_attention_gqa_dpas, params);
        } else if (dpas_attention) {
            add_stage(sparse_attention_dpas, params);
        }
        // Whether a given decode call is long enough to route through the split path is
        // runtime-varying (context grows every step), so both split kernels are always
        // compiled and gated purely at execute() time via QSARuntimeParams::use_split_attention.
        add_stage(sparse_attention_split, params);
        add_stage(sparse_attention_split_reduce, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<QSACmImpl>(this);
    }

    // Recomputed rather than cached: make_deep_copy() rebuilds the impl with the default
    // constructor and copies only _order and the per-stage kernel data.
    static bool use_dpas_score(const kernel_impl_params& params) {
        return can_use_dpas_score(*params.typed_desc<qsa>(), params.get_device_info());
    }
    static bool use_dpas_attention(const kernel_impl_params& params) {
        return can_use_dpas_attention(*params.typed_desc<qsa>(), params.get_device_info());
    }
    static bool use_dpas_projection(const kernel_impl_params& params) {
        return can_use_dpas_projection(*params.typed_desc<qsa>(), params.get_device_info());
    }
    static bool use_gqa_dpas_attention(const kernel_impl_params& params) {
        return can_use_gqa_dpas_attention(*params.typed_desc<qsa>(), params.get_device_info());
    }

    // Runtime state must be refreshed before the base sizes the internal buffers.
    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        update_rt_params(inst);
        PrimitiveImplCM::update(inst, impl_params);
    }

    [[nodiscard]] bool requires_update(primitive_inst& /*inst*/, const kernel_impl_params& /*impl_params*/) const override {
        // past_lens / subsequence_begins change every iteration even when shapes do not.
        return true;
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        const auto desc = params.typed_desc<qsa>();

        // Sizes are derived from the static input layouts, never from the runtime state:
        // internal buffers are allocated once per shape, while past_lens / subsequence_begins
        // change every iteration, so a runtime-derived size would under-allocate.
        const auto tokens = std::max<size_t>(1, params.get_input_layout(QSAIn::QUERY).get_shape()[0]);
        const auto seqs = std::max<size_t>(1, params.get_input_layout(QSAIn::PAST_LENS).get_shape()[0]);

        // Upper bound on visible QSA blocks: the whole paged cache divided by the ratio.
        const auto& kcache = params.get_input_layout(QSAIn::KEY_CACHE).get_shape();
        const auto max_blocks = block_scores_stride(std::max<size_t>(1, kcache[0] * desc->pa_block_size / desc->compress_ratio));

        // A sequence touches at most ceil(q_len / r) + 1 QSA blocks, so tokens + seqs pairs
        // is a safe bound for the Q1 work-group map.
        const auto prepare_pairs = tokens + seqs;

        // Bounded query-row capacity of BLOCK_SCORES (see score_scratch_rows()): every
        // chunk boundary can also split what would otherwise be one 16-query tile into
        // two (tiles are re-anchored at max(chunk_begin, sequence_begin)), on top of the
        // existing one-tile-per-sequence-boundary bound, so the number of chunks is added
        // to the tile-count upper bound as well.
        const auto scratch_rows = score_scratch_rows(params, *desc);
        const auto num_chunks = std::max<size_t>(1, (tokens + scratch_rows - 1) / scratch_rows);

        // One 16-query tile per sequence-local or chunk-local boundary: at most
        // ceil(tokens/16) + seqs + num_chunks tiles.
        const auto score_tiles = tokens / dpas_query_tile + seqs + num_chunks + 1;
        const auto proj_width = (desc->indexer_heads + 1) * desc->indexer_head_dim;

        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(prepare_pairs * 2, ov::element::i32, true, false);                         // PREPARE_MAP (host-filled)
        bufs.emplace_back(tokens * desc->indexer_heads * desc->indexer_head_dim, ov::element::f16);  // INDEXER_Q
        bufs.emplace_back(scratch_rows * max_blocks, ov::element::f32);                              // BLOCK_SCORES (bounded)
        bufs.emplace_back(tokens * desc->block_topk, ov::element::i32);                              // SELECTED_BLOCKS
        bufs.emplace_back(tokens, ov::element::i32);                                                 // SELECTED_COUNTS
        bufs.emplace_back(score_tiles * 2, ov::element::i32, true, false);                           // SCORE_TILE_MAP (host-filled)
        bufs.emplace_back(tokens * proj_width, ov::element::f32);                                    // PROJECTED (Q1 dpas)
        bufs.emplace_back(tokens * 2, ov::element::i32, true, false);                                // TOKEN_MAP (host-filled)
        // SPLIT_ACC/SPLIT_ML are only ever dispatched when every sequence contributes exactly
        // one new token (num_tokens == num_seqs, enforced at execute() time by
        // QSARuntimeParams::use_split_attention), so sizing them by `seqs` safely bounds
        // them even though `tokens` can be far larger for a prefill-shaped specialization
        // of the same primitive.
        bufs.emplace_back(seqs * desc->heads_num * split_partitions * desc->v_head_size, ov::element::f32);  // SPLIT_ACC
        bufs.emplace_back(seqs * desc->heads_num * split_partitions * 2, ov::element::f32);                  // SPLIT_ML
        return bufs;
    }

    // execute_stage() only calls a stage's DispatchDataFunc / re-binds its kernel arguments
    // when kd.need_dispatch_data_update / need_args_update are true, then clears them; it
    // assumes a stage is dispatched at most once per execute() call. The Q2 chunk loop
    // below calls the SAME score/topk stages multiple times in one execute() call with
    // DIFFERENT rtp->chunk_* values each time, so those flags must be forced true again
    // before every chunk's call, or chunks after the first would silently redispatch with
    // chunk 0's stale scalars/work-group sizes.
    static void force_stage_refresh(Stage::Ptr& stage) {
        stage->kd.need_dispatch_data_update = true;
        stage->kd.need_args_update = true;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        update_rt_params(instance);
        auto* rtp = &rt();
        const auto& params = *instance.get_impl_params();

        g_qsa_stage_events.clear();
        upload_prepare_map(instance);

        // Q0 and Q1 are mutually independent (the indexer never reads the main KV and the
        // main KV write never reads indexer state), so they are enqueued back to back on
        // the same set of input events instead of being chained.
        auto kv_event = execute_stage(events, instance, kv_cache_update);
        record_stage_event("Q0_kv_cache_update", kv_event);

        // Q1: the systolic projection (when supported) must complete before qsa_prepare
        // reads its FP32 output; when unsupported, qsa_prepare stays independent of any
        // prior stage exactly as before (its own scalar-tiled projection is unaffected).
        std::vector<event::ptr> prepare_deps = events;
        if (use_dpas_projection(params)) {
            auto proj_event = execute_stage(events, instance, project_dpas);
            record_stage_event("Q1_project", proj_event);
            prepare_deps = {proj_event};
        }
        auto prepare_event = execute_stage(prepare_deps, instance, prepare);
        record_stage_event("Q1_prepare", prepare_event);

        std::vector<event::ptr> after_prepare = {prepare_event};
        std::vector<event::ptr> after_select;
        if (rtp->stage == QSAStage::GENERATE || use_dpas_score(params)) {
            // Decode (score_partition) and prefill/mixed dpas scoring (score_tile_dpas)
            // both run as a sequence of bounded query-row chunks against the SAME reused
            // BLOCK_SCORES scratch: chunk i+1's score dispatch must not start before chunk
            // i's topk_radix_wg has finished reading it, so chunks are chained strictly
            // sequentially (never run concurrently). A shape that fits the score budget in
            // one chunk (the common case) just runs this loop once, identical to a single
            // unchunked call.
            std::vector<event::ptr> chunk_deps = after_prepare;
            for (const auto& chunk : rtp->score_chunks) {
                rtp->chunk_row_begin = chunk.row_begin;
                rtp->chunk_row_end = chunk.row_end;
                rtp->chunk_tile_offset = chunk.tile_offset;
                rtp->chunk_tile_count = chunk.tile_count;
                event::ptr scored;
                if (rtp->stage == QSAStage::GENERATE) {
                    // Decode: partition along the block range, then merge with the
                    // cooperative fast-wg16-cached exact radix select.
                    force_stage_refresh(score_partition);
                    scored = execute_stage(chunk_deps, instance, score_partition);
                } else {
                    // Prefill / mixed: the query axis fills the dpas N axis, so scoring
                    // runs on the systolic array; selection reuses the decode
                    // finalization kernel so that both paths select identically and no
                    // work-group reads back its peers' scores.
                    force_stage_refresh(score_tile_dpas);
                    scored = execute_stage(chunk_deps, instance, score_tile_dpas);
                }
                record_stage_event("Q2_score", scored);
                force_stage_refresh(topk_radix_wg);
                auto fin = execute_stage({scored}, instance, topk_radix_wg);
                record_stage_event("Q2_topk_finalization", fin);
                chunk_deps = {fin};
            }
            after_select = chunk_deps;
        } else {
            // Same maths without cm_dpas: Xe1, or a shape the tiled path cannot cover. Score
            // and top-k are fused into a single kernel here, so there is no separate
            // Q2_topk_finalization event to record for this path. Not row-chunk aware
            // (BLOCK_SCORES is sized unbounded for this device via score_scratch_rows()).
            auto fused = execute_stage(after_prepare, instance, score_topk_fused);
            record_stage_event("Q2_score", fused);
            after_select = {fused};
        }

        // Q3 waits on both Q0 (main KV visible) and Q2 (selected metadata ready).
        std::vector<event::ptr> q3_deps = {kv_event, after_select[0]};
        if (rtp->stage == QSAStage::GENERATE) {
            if (rtp->use_split_attention) {
                // Long decode: split each token's selected KV across split_partitions
                // independent partial-softmax producers, then merge exactly.
                auto split_ev = execute_stage(q3_deps, instance, sparse_attention_split);
                record_stage_event("Q3_split", split_ev);
                auto reduce_ev = execute_stage({split_ev}, instance, sparse_attention_split_reduce);
                record_stage_event("Q3_reduce", reduce_ev);
                return reduce_ev;
            }
            auto ev = execute_stage(q3_deps, instance, sparse_attention);
            record_stage_event("Q3_sparse_attention", ev);
            return ev;
        }
        // The dpas variants need a real query tile to fill the dpas N axis, so they are
        // used for prefill/mixed only; decode is handled above (scalar or split-selected-KV).
        if (use_gqa_dpas_attention(params)) {
            auto ev = execute_stage(q3_deps, instance, sparse_attention_gqa_dpas);
            record_stage_event("Q3_sparse_attention", ev);
            return ev;
        }
        if (use_dpas_attention(params)) {
            auto ev = execute_stage(q3_deps, instance, sparse_attention_dpas);
            record_stage_event("Q3_sparse_attention", ev);
            return ev;
        }
        auto ev = execute_stage(q3_deps, instance, sparse_attention);
        record_stage_event("Q3_sparse_attention", ev);
        return ev;
    }

public:
    // Reads past_lens / subsequence_begins from the instance (not from kernel_impl_params:
    // memory_deps is not populated for a static-shape node) and derives everything the
    // stages need: token/sequence counts, the Q1 (sequence, QSA block) work-group map,
    // the visible block count and the decode partition count.
    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        auto* rtp = &rt();

        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<qsa>();
        const auto past_lens = read_i32_input(instance, QSAIn::PAST_LENS);
        const auto subsequence_begins = read_i32_input(instance, QSAIn::SUBSEQUENCE_BEGINS);

        const size_t num_seqs = past_lens.size();
        OPENVINO_ASSERT(subsequence_begins.size() == num_seqs + 1, "[GPU][CM] qsa: subsequence_begins must have num_seqs + 1 entries");

        rtp->num_seqs = num_seqs;
        rtp->num_tokens = num_seqs == 0 ? 0 : static_cast<size_t>(subsequence_begins.back());

        const auto r = desc->compress_ratio;

        bool any_multi = false;
        bool any_single = false;
        size_t max_blocks = 1;
        size_t min_past_len = std::numeric_limits<size_t>::max();
        rtp->prepare_map.clear();
        rtp->score_tile_map.clear();
        rtp->token_map.clear();

        for (size_t s = 0; s < num_seqs; s++) {
            const auto q_len = static_cast<size_t>(std::max(0, subsequence_begins[s + 1] - subsequence_begins[s]));
            const auto past_len = static_cast<size_t>(std::max(0, past_lens[s]));
            if (q_len == 0) {
                continue;
            }
            (q_len > 1) ? (any_multi = true) : (any_single = true);
            min_past_len = std::min(min_past_len, past_len);

            const size_t total_len = past_len + q_len;
            max_blocks = std::max(max_blocks, total_len / r);

            const auto seq_begin = static_cast<size_t>(subsequence_begins[s]);

            // One (sequence, flat token) entry per token, in flat-token order, for the GQA
            // dpas Q3's per-token dispatch (qsa_sparse_attention_gqa_dpas.cm's token_map).
            for (size_t t = 0; t < q_len; t++) {
                rtp->token_map.push_back(static_cast<int32_t>(s));
                rtp->token_map.push_back(static_cast<int32_t>(seq_begin + t));
            }

            // Every QSA block touched by the new tokens becomes one Q1 work-group.
            const size_t first_block = past_len / r;
            const size_t last_block = (total_len - 1) / r;
            for (size_t b = first_block; b <= last_block; b++) {
                rtp->prepare_map.push_back(static_cast<int32_t>(s));
                rtp->prepare_map.push_back(static_cast<int32_t>(b));
            }
        }

        // Bounded-scratch query-row chunk plan + its score_tile_map: chunk boundaries are
        // plain row-count windows over the flat 0..num_tokens range (independent of
        // sequence boundaries), and within each chunk, 16-query tiles are re-anchored at
        // max(chunk_begin, sequence_begin) so a tile never straddles a chunk OR a
        // sequence edge. The row-chunk-aware kernels clamp `valid`/`row_end - token0`
        // themselves, so a tile may legitimately be shorter than 16 at either kind of
        // edge. Uses the SAME static-shape-derived capacity as get_internal_buffer_descs()
        // (score_scratch_rows), so the chunk loop in execute() never exceeds the
        // allocated BLOCK_SCORES buffer. Always produces >= 1 chunk, even for
        // num_tokens == 0, matching every dispatch function's std::max<size_t>(1, ...)
        // convention elsewhere.
        rtp->score_chunks.clear();
        {
            const size_t scratch_rows = std::max<size_t>(1, score_scratch_rows(params, *desc));
            size_t seq = 0;
            size_t begin = 0;
            do {
                const size_t end = std::min(rtp->num_tokens, begin + scratch_rows);
                const size_t tile_offset = rtp->score_tile_map.size() / 2;
                while (seq + 1 < subsequence_begins.size() &&
                       static_cast<size_t>(subsequence_begins[seq + 1]) <= begin) {
                    seq++;
                }
                size_t s = seq;
                while (s + 1 < subsequence_begins.size() && static_cast<size_t>(subsequence_begins[s]) < end) {
                    const size_t s_begin = static_cast<size_t>(subsequence_begins[s]);
                    const size_t s_end = static_cast<size_t>(subsequence_begins[s + 1]);
                    const size_t tile_start = std::max(begin, s_begin);
                    const size_t tile_end = std::min(end, s_end);
                    for (size_t t = tile_start; t < tile_end; t += dpas_query_tile) {
                        rtp->score_tile_map.push_back(static_cast<int32_t>(s));
                        rtp->score_tile_map.push_back(static_cast<int32_t>(t));
                    }
                    s++;
                }
                rtp->score_chunks.push_back(
                    QSAScoreChunk{begin, end, tile_offset, rtp->score_tile_map.size() / 2 - tile_offset});
                begin = end;
            } while (begin < rtp->num_tokens);
        }

        rtp->prepare_wg_count = rtp->prepare_map.size() / 2;
        rtp->score_tile_count = rtp->score_tile_map.size() / 2;
        rtp->max_blocks = block_scores_stride(max_blocks);
        rtp->num_partitions = std::max<size_t>(1, (rtp->max_blocks + partition_blocks - 1) / partition_blocks);
        rtp->stage = (any_multi && any_single) ? QSAStage::MIXED : (any_multi ? QSAStage::PREFILL : QSAStage::GENERATE);
        // Only pure decode (every sequence contributes exactly one new token) with enough
        // history for the block-selection budget to matter routes through the
        // split-selected-KV Q3; short decode is already cheap on the scalar kernel. This
        // is independent of batch size since neither kernel's dispatch grid depends on
        // the number of sequences.
        rtp->use_split_attention = rtp->stage == QSAStage::GENERATE && any_single &&
                                   min_past_len != std::numeric_limits<size_t>::max() &&
                                   min_past_len >= split_decode_min_past;
    }

private:

    static void upload_map(primitive_inst& instance, size_t buf_idx, const std::vector<int32_t>& data, const char* what) {
        if (data.empty()) {
            return;
        }
        auto mem = instance.get_intermediates_memories()[buf_idx];
        OPENVINO_ASSERT(mem->count() >= data.size(), "[GPU][CM] qsa ", what, " buffer is too small");
        mem_lock<int32_t, mem_lock_type::write> lock(mem, instance.get_network().get_stream());
        std::copy(data.begin(), data.end(), lock.begin());
    }

    void upload_prepare_map(primitive_inst& instance) {
        auto* rtp = &rt();
        upload_map(instance, QSAInternalBufIdx::PREPARE_MAP, rtp->prepare_map, "prepare map");
        const auto& params = *instance.get_impl_params();
        if (use_dpas_score(params) || use_dpas_attention(params)) {
            upload_map(instance, QSAInternalBufIdx::SCORE_TILE_MAP, rtp->score_tile_map, "score tile map");
        }
        if (use_gqa_dpas_attention(params)) {
            upload_map(instance, QSAInternalBufIdx::TOKEN_MAP, rtp->token_map, "token map");
        }
    }
};

}  // namespace

namespace qsa_debug {

std::vector<std::pair<std::string, event::ptr>> take_stage_events() {
    std::vector<std::pair<std::string, event::ptr>> result;
    result.swap(g_qsa_stage_events);
    return result;
}

}  // namespace qsa_debug

bool QSACMImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::qsa>());

    if (node.has_fused_primitives()) {
        return false;
    }

    auto& engine = node.get_program().get_engine();
    const auto& config = node.get_program().get_config();
    const auto& info = engine.get_device_info();
    if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
        return false;
    }

    const auto desc = node.as<cldnn::qsa>().get_primitive();
    std::string reason;
    if (!ov::intel_gpu::op::QSA::validate_config(desc->get_config(), reason)) {
        GPU_DEBUG_TRACE_DETAIL << "[GPU][CM] qsa validate_impl() - false: " << reason << std::endl;
        return false;
    }

    // First version: f16-only main KV cache, no compression scales / zero points.
    static constexpr std::array f16_inputs = {QSAIn::MIXED,
                                              QSAIn::QUERY,
                                              QSAIn::KEY,
                                              QSAIn::VALUE,
                                              QSAIn::KEY_CACHE,
                                              QSAIn::VALUE_CACHE,
                                              QSAIn::INDEXER_RAW_K_CACHE,
                                              QSAIn::INDEXER_SUMMARY_CACHE,
                                              QSAIn::INDEXER_QK_WEIGHT,
                                              QSAIn::INDEXER_Q_NORM_WEIGHT,
                                              QSAIn::INDEXER_K_NORM_WEIGHT,
                                              QSAIn::ROTARY_COS_SIN};
    for (auto idx : f16_inputs) {
        const auto& l = node.get_input_layout(idx);
        if (l.data_type != ov::element::f16 || l.format != cldnn::format::bfyx) {
            return false;
        }
    }

    static constexpr std::array i32_inputs = {QSAIn::PAST_LENS,
                                              QSAIn::SUBSEQUENCE_BEGINS,
                                              QSAIn::BLOCK_INDICES,
                                              QSAIn::BLOCK_INDICES_BEGINS,
                                              QSAIn::POSITION_IDS};
    for (auto idx : i32_inputs) {
        if (node.get_input_layout(idx).data_type != ov::element::i32) {
            return false;
        }
    }

    const auto& out = node.get_output_layout(0);
    return out.data_type == ov::element::f16 && out.format == cldnn::format::bfyx;
}

std::unique_ptr<cldnn::primitive_impl> QSACMImplementationManager::create_impl(const cldnn::program_node& node,
                                                                              const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::qsa>());
    return std::make_unique<QSACmImpl>(params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::qsa)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::QSACmImpl)

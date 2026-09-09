// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "../primitive_cm_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/qsa.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "primitive_inst.h"
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
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
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
        for (uint32_t i = 0; i < 2; i++) {
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
            wgs.global = {std::max<size_t>(1, rtp->score_tile_count) * score_wg_size, 1, 1};
            wgs.local = {score_wg_size, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->score_tile_count);
            push_u32(kd, rtp->max_blocks);
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
            // Partitions run along the block range only. Splitting along Hidx is forbidden
            // because ReLU precedes the head reduction.
            wgs.global = {std::max<size_t>(1, rtp->num_tokens), 1, std::max<size_t>(1, rtp->num_partitions)};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, rtp->max_blocks);
        }};
    }
};

// ===========================================================================
// Q2 decode pass 2 - top-k finalization
// ===========================================================================
class QSATopkFinalizationGenerator : public KernelGenerator {
public:
    QSATopkFinalizationGenerator() : KernelGenerator("qsa_topk_finalization", "_cm") {}

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
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAInternalBufIdx::BLOCK_SCORES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
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

// ---------------------------------------------------------------------------

class QSACmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::QSACmImpl)

    Stage::Ptr kv_cache_update = make_stage<QSAKVCacheUpdateGenerator>();
    Stage::Ptr prepare = make_stage<QSAPrepareGenerator>();
    Stage::Ptr score_topk_fused = make_stage<QSAScoreTopkFusedGenerator>();
    Stage::Ptr score_tile_dpas = make_stage<QSAScoreTileDpasGenerator>();
    Stage::Ptr score_partition = make_stage<QSAScorePartitionGenerator>();
    Stage::Ptr topk_finalization = make_stage<QSATopkFinalizationGenerator>();
    Stage::Ptr sparse_attention = make_stage<QSASparseAttentionGenerator>();
    Stage::Ptr sparse_attention_dpas = make_stage<QSASparseAttentionDpasGenerator>();

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
        add_stage(kv_cache_update, params);
        add_stage(prepare, params);
        add_stage(score_topk_fused, params);
        if (dpas_score) {
            add_stage(score_tile_dpas, params);
        }
        add_stage(score_partition, params);
        add_stage(topk_finalization, params);
        add_stage(sparse_attention, params);
        if (dpas_attention) {
            add_stage(sparse_attention_dpas, params);
        }
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

        // One 16-query tile per sequence-local chunk: at most ceil(tokens/16) + seqs tiles.
        const auto score_tiles = tokens / dpas_query_tile + seqs + 1;

        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(prepare_pairs * 2, ov::element::i32, true, false);                         // PREPARE_MAP (host-filled)
        bufs.emplace_back(tokens * desc->indexer_heads * desc->indexer_head_dim, ov::element::f16);  // INDEXER_Q
        bufs.emplace_back(tokens * max_blocks, ov::element::f32);                                    // BLOCK_SCORES
        bufs.emplace_back(tokens * desc->block_topk, ov::element::i32);                              // SELECTED_BLOCKS
        bufs.emplace_back(tokens, ov::element::i32);                                                 // SELECTED_COUNTS
        bufs.emplace_back(score_tiles * 2, ov::element::i32, true, false);                           // SCORE_TILE_MAP (host-filled)
        return bufs;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        update_rt_params(instance);
        auto* rtp = &rt();
        const auto& params = *instance.get_impl_params();

        upload_prepare_map(instance);

        // Q0 and Q1 are mutually independent (the indexer never reads the main KV and the
        // main KV write never reads indexer state), so they are enqueued back to back on
        // the same set of input events instead of being chained.
        auto kv_event = execute_stage(events, instance, kv_cache_update);
        auto prepare_event = execute_stage(events, instance, prepare);

        std::vector<event::ptr> after_prepare = {prepare_event};
        std::vector<event::ptr> after_select;
        if (rtp->stage == QSAStage::GENERATE) {
            // Decode: partition along the block range, then merge with an exact radix select.
            auto part = execute_stage(after_prepare, instance, score_partition);
            after_select = {execute_stage({part}, instance, topk_finalization)};
        } else if (use_dpas_score(params)) {
            // Prefill / mixed: the query axis fills the dpas N axis, so scoring runs on the
            // systolic array; selection reuses the decode finalization kernel so that both
            // paths select identically and no work-group reads back its peers' scores.
            auto scored = execute_stage(after_prepare, instance, score_tile_dpas);
            after_select = {execute_stage({scored}, instance, topk_finalization)};
        } else {
            // Same maths without cm_dpas: Xe1, or a shape the tiled path cannot cover.
            after_select = {execute_stage(after_prepare, instance, score_topk_fused)};
        }

        // Q3 waits on both Q0 (main KV visible) and Q2 (selected metadata ready).
        std::vector<event::ptr> q3_deps = {kv_event, after_select[0]};
        // The query-tile variant needs a real query tile to fill the dpas N axis, so it is
        // used for prefill/mixed only; decode keeps the per-query scalar kernel.
        if (use_dpas_attention(params) && rtp->stage != QSAStage::GENERATE) {
            return execute_stage(q3_deps, instance, sparse_attention_dpas);
        }
        return execute_stage(q3_deps, instance, sparse_attention);
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
        rtp->prepare_map.clear();
        rtp->score_tile_map.clear();

        for (size_t s = 0; s < num_seqs; s++) {
            const auto q_len = static_cast<size_t>(std::max(0, subsequence_begins[s + 1] - subsequence_begins[s]));
            const auto past_len = static_cast<size_t>(std::max(0, past_lens[s]));
            if (q_len == 0) {
                continue;
            }
            (q_len > 1) ? (any_multi = true) : (any_single = true);

            const size_t total_len = past_len + q_len;
            max_blocks = std::max(max_blocks, total_len / r);

            // A prefill scoring tile is 16 queries of ONE sequence: page tables and
            // past_lens differ per sequence, so a tile must never straddle a boundary.
            for (size_t t = 0; t < q_len; t += dpas_query_tile) {
                rtp->score_tile_map.push_back(static_cast<int32_t>(s));
                rtp->score_tile_map.push_back(static_cast<int32_t>(subsequence_begins[s]) + static_cast<int32_t>(t));
            }

            // Every QSA block touched by the new tokens becomes one Q1 work-group.
            const size_t first_block = past_len / r;
            const size_t last_block = (total_len - 1) / r;
            for (size_t b = first_block; b <= last_block; b++) {
                rtp->prepare_map.push_back(static_cast<int32_t>(s));
                rtp->prepare_map.push_back(static_cast<int32_t>(b));
            }
        }

        rtp->prepare_wg_count = rtp->prepare_map.size() / 2;
        rtp->score_tile_count = rtp->score_tile_map.size() / 2;
        rtp->max_blocks = block_scores_stride(max_blocks);
        rtp->num_partitions = std::max<size_t>(1, (rtp->max_blocks + partition_blocks - 1) / partition_blocks);
        rtp->stage = (any_multi && any_single) ? QSAStage::MIXED : (any_multi ? QSAStage::PREFILL : QSAStage::GENERATE);
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
    }
};

}  // namespace

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

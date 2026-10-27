// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../primitive_cm_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "../utils/qsa_host_common.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/qsa_indexer.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::cm {

using namespace cldnn;
using QSAIn = ov::intel_gpu::op::QSAIndexer::InputIdx;
using QSAOut = ov::intel_gpu::op::QSAIndexer::OutputIdx;

namespace {

QSAIndexerRuntimeParams* rt_of(ImplRuntimeParams* rt) {
    auto* p = static_cast<QSAIndexerRuntimeParams*>(rt);
    OPENVINO_ASSERT(p != nullptr, "[GPU][CM] qsa_indexer runtime params are not initialized");
    return p;
}

/// H_iq / H_ik, derived from the 4D [B, L, H, D_idx] index_query/raw_index_keys layouts (B/L may be dynamic).
size_t index_heads_4d(const layout& l) {
    const auto& ps = l.get_partial_shape();
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() >= 3 && ps[2].is_static(),
                    "[GPU][CM] qsa_indexer: expected a rank-4 [B, L, H, D_idx] input with a static head count");
    return static_cast<size_t>(ps[2].get_length());
}

JitConstants make_jit(const std::shared_ptr<const qsa_indexer>& desc, const RuntimeParams& params) {
    const auto h_iq = index_heads_4d(params.get_input_layout(QSAIn::INDEX_QUERY));

    JitConstants jit;
    jit.make("QSA_IDX_QUERY_HEADS", h_iq);
    jit.make("QSA_IDX_HEAD_DIM", desc->indexer_head_dim);
    jit.make("QSA_IDX_ROTARY_DIM", desc->indexer_rotary_dim);
    jit.make("QSA_COMPRESS_RATIO", desc->key_block_size);
    jit.make("QSA_BLOCK_TOPK", desc->block_topk);
    jit.make("QSA_ADDRESS_MODE", 1);
    jit.make("QSA_PREPARE_WG", h_iq + 1);
    jit.add(make_jit_constant("QSA_IDX_NORM_EPS", desc->epsilon));
    jit.add(make_jit_constant("QSA_IDX_SCALE", ov::intel_gpu::op::QSAIndexer::effective_scale(desc->get_config())));
    return jit;
}

constexpr auto get_qsa_indexer_build_options() {
    return qsa_cm_build_options;
}

// qsa_indexer_topk's cooperative work-group size (see qsa_topk_radix_wg.cm's proven choice).

constexpr size_t max_slm_bytes = 64 * 1024;
constexpr size_t score_wg_size = 16;  // threads per query tile in the dpas Q2 score kernel

// Decode-only Q2 score-partition kernel: partitions each token's block range so a single
// decode query gets real parallelism. 32 blocks per work-item was fastest on BMG for
// Qwen3.8 decode at 4K-32K context (512 was ~6-15x slower there).
constexpr size_t score_partition_blocks = 32;

/// Whether every batch row in this call has exactly one query token. A shape-agnostic
/// impl keeps both the decode and prefill stages and decides per call.
bool is_decode_shape(const kernel_impl_params& params) {
    const auto& index_query_layout = params.get_input_layout(QSAIn::INDEX_QUERY);
    if (index_query_layout.is_dynamic()) {
        return false;
    }
    const auto& shape = index_query_layout.get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "[GPU][CM] qsa_indexer: expected a rank-4 [B, L, H, D_idx] input");
    return shape[1] == 1;
}

bool may_decode(const kernel_impl_params& params) {
    return params.is_dynamic() || is_decode_shape(params);
}

bool may_prefill(const kernel_impl_params& params) {
    return params.is_dynamic() || !is_decode_shape(params);
}

/// f32 element capacity of BLOCK_SCORES. Static shapes only, so allocation and per-call
/// chunk planning always agree. Only decode (score_partition + topk) is row-chunk aware.
size_t score_scratch_elems(const kernel_impl_params& params, const qsa_indexer& desc) {
    const auto& index_query_shape = params.get_input_layout(QSAIn::INDEX_QUERY).get_shape();
    const auto tokens = std::max<size_t>(1, index_query_shape[0] * index_query_shape[1]);
    const auto& summary_shape = params.get_input_layout(QSAIn::BLOCK_INDEX_KEYS_CACHE).get_shape();
    const auto max_blocks = std::max<size_t>(1, summary_shape[2]);
    if (!is_decode_shape(params)) {
        return tokens * max_blocks;
    }
    const auto budget_rows = std::max<size_t>(1, score_budget_bytes() / (max_blocks * sizeof(float)));
    return std::min(tokens, budget_rows) * max_blocks;
}

/// Whether the systolic qsa_indexer_score_dpas kernel can be used. Dense addressing has no
/// page table, so (unlike the paged capability gate) there is no page/tile alignment
/// constraint at all - any consecutive run of summaries within one batch row is already
/// contiguous.
bool can_use_dpas_score(const qsa_indexer& desc, size_t h_iq, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;  // REG_N is 8 on Xe1, the kernel assumes a 16-wide N axis
    }
    if (desc.indexer_head_dim % dpas_k_step != 0) {
        return false;
    }
    if (desc.indexer_head_dim * sizeof(ov::float16) < 64) {
        return false;  // 2D block load surface width minimum
    }
    const auto staged_bytes = h_iq * (desc.indexer_head_dim / dpas_k_step) * dpas_k_step * dpas_query_tile * sizeof(ov::float16);
    return staged_bytes <= max_slm_bytes;
}

// ===========================================================================
// Q1 - indexer prepare (dense addressing)
// ===========================================================================
class DensePrepareGenerator : public KernelGenerator {
public:
    DensePrepareGenerator() : KernelGenerator("qsa_indexer_prepare_dense", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<qsa_indexer>(), params));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEX_QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::RAW_INDEX_KEYS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::Q_NORM_WEIGHT});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::K_NORM_WEIGHT});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::ROPE_COS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::ROPE_SIN});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::KEY_LENGTH});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_INCOMPLETE_RAW_K});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_INCOMPLETE_ROPE});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, QSAOut::BLOCK_INDEX_KEYS_CACHE});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, QSAOut::PAST_INCOMPLETE_RAW_K});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, QSAOut::PAST_INCOMPLETE_ROPE});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::INDEXER_Q});
        for (uint32_t i = 0; i < 4; i++) {
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
            const auto wg = rtp->index_query_heads + 1;
            const size_t wg_count = std::max<size_t>(1, rtp->prepare_wg_count);

            auto& wgs = kd.params.workGroups;
            wgs.global = {1, 1, wg_count * wg};
            wgs.local = {1, 1, wg};

            kd.params.scalars.clear();
            push_u32(kd, rtp->prepare_wg_count);
            push_u32(kd, rtp->prepare_blocks_per_row);
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->summary_stride);
        }};
    }
};

// ===========================================================================
// Q2 prefill, systolic variant - score (dense addressing) on cm_dpas
// ===========================================================================
class DenseScoreDpasGenerator : public KernelGenerator {
public:
    DenseScoreDpasGenerator() : KernelGenerator("qsa_indexer_score_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<qsa_indexer>(), params));
        jit.make("QSA_SCORE_WG", score_wg_size);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, QSAOut::BLOCK_INDEX_KEYS_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::KEY_LENGTH});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::BLOCK_SCORES});
        for (uint32_t i = 0; i < 4; i++) {
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
            // One work-group owns one 16-query tile.
            wgs.global = {std::max<size_t>(1, rtp->score_tile_count) * score_wg_size, 1, 1};
            wgs.local = {score_wg_size, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->score_tile_count);
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->summary_stride);
            push_u32(kd, rtp->max_blocks);
        }};
    }
};

// ===========================================================================
// Q2 - top-k finalize (dense addressing); writes directly to the op's own outputs
// ===========================================================================
class DenseTopkGenerator : public KernelGenerator {
public:
    explicit DenseTopkGenerator(bool decode = false) : KernelGenerator("qsa_indexer_topk", decode ? "_decode_cm" : "_cm"), m_decode(decode) {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<qsa_indexer>(), params));
        jit.make("QSA_TOPK_WG", topk_wg_size(m_decode));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::BLOCK_SCORES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::KEY_LENGTH});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
        for (uint32_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[decode = m_decode](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            auto* rtp = rt_of(rt);
            auto& wgs = kd.params.workGroups;
            // One work-group cooperatively finalizes one token's row.
            const auto wg = topk_wg_size(decode);
            wgs.global = {std::max<size_t>(1, rtp->chunk_row_end - rtp->chunk_row_begin) * wg, 1, 1};
            wgs.local = {wg, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->max_blocks);
            push_u32(kd, rtp->chunk_row_begin);
            push_u32(kd, rtp->chunk_row_end);
        }};
    }

private:
    bool m_decode;
};

// ===========================================================================
// Q2 prefill/mixed without dpas - fused score + top-k (dense addressing)
// ===========================================================================
class DenseScoreTopkFusedGenerator : public KernelGenerator {
public:
    DenseScoreTopkFusedGenerator() : KernelGenerator("qsa_indexer_score_topk_fused", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<qsa_indexer>(), params));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, QSAOut::BLOCK_INDEX_KEYS_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::KEY_LENGTH});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::BLOCK_SCORES});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
        for (uint32_t i = 0; i < 4; i++) {
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
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->summary_stride);
            push_u32(kd, rtp->max_blocks);
        }};
    }
};

// ===========================================================================
// Q2 decode, score-partition variant (dense addressing)
// ===========================================================================
class DenseScorePartitionGenerator : public KernelGenerator {
public:
    DenseScorePartitionGenerator() : KernelGenerator("qsa_indexer_score_partition", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<qsa_indexer>(), params));
        jit.make("QSA_SCORE_PARTITION_BLOCKS", score_partition_blocks);
        // Rows whose blocks are all selected are never read by qsa_indexer_topk.
        jit.make("QSA_DENSE_SCORE_BYPASS", 1);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, QSAOut::BLOCK_INDEX_KEYS_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::KEY_LENGTH});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::BLOCK_SCORES});
        for (uint32_t i = 0; i < 7; i++) {
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
            wgs.global = {std::max<size_t>(1, rtp->chunk_row_end - rtp->chunk_row_begin), 1, rtp->num_partitions};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->summary_stride);
            push_u32(kd, rtp->max_blocks);
            push_u32(kd, rtp->num_partitions);
            push_u32(kd, rtp->chunk_row_begin);
            push_u32(kd, rtp->chunk_row_end);
        }};
    }
};

class QSAIndexerCmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::QSAIndexerCmImpl)

    Stage::Ptr prepare = make_stage<DensePrepareGenerator>();
    Stage::Ptr score_topk_fused = make_stage<DenseScoreTopkFusedGenerator>();
    Stage::Ptr score_dpas = make_stage<DenseScoreDpasGenerator>();
    Stage::Ptr score_partition = make_stage<DenseScorePartitionGenerator>();
    Stage::Ptr topk = make_stage<DenseTopkGenerator>();
    Stage::Ptr topk_decode = make_stage<DenseTopkGenerator>(true);

    QSAIndexerCmImpl() : PrimitiveImplCM(QSAIndexerCMImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<QSAIndexerRuntimeParams>();
    }

    QSAIndexerRuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<QSAIndexerRuntimeParams>();
        }
        return *static_cast<QSAIndexerRuntimeParams*>(m_rt_params.get());
    }

    // Q2: decode -> score_partition + topk (row-chunked); prefill/mixed -> score_dpas + topk,
    // or the fused score+top-k kernel when dpas is unavailable.
    explicit QSAIndexerCmImpl(const RuntimeParams& params) : QSAIndexerCmImpl() {
        add_stage(prepare, params);
        if (may_decode(params)) {
            add_stage(score_partition, params);
            add_stage(topk_decode, params);
        }
        if (may_prefill(params)) {
            if (use_dpas_score(params)) {
                add_stage(score_dpas, params);
                add_stage(topk, params);
            } else {
                add_stage(score_topk_fused, params);
            }
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<QSAIndexerCmImpl>(this);
    }

    [[nodiscard]] bool requires_update(primitive_inst& /*inst*/, const kernel_impl_params& /*impl_params*/) const override {
        return true;
    }

    // Recomputed rather than cached: make_deep_copy() rebuilds the impl with the default
    // constructor and copies only _order and the per-stage kernel data.
    static bool use_dpas_score(const kernel_impl_params& params) {
        const auto desc = params.typed_desc<qsa_indexer>();
        const auto h_iq = index_heads_4d(params.get_input_layout(QSAIn::INDEX_QUERY));
        return can_use_dpas_score(*desc, h_iq, params.get_device_info());
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        if (params.is_dynamic()) {
            return {};
        }
        const auto desc = params.typed_desc<qsa_indexer>();
        const auto& index_query_shape = params.get_input_layout(QSAIn::INDEX_QUERY).get_shape();
        const auto batch = std::max<size_t>(1, index_query_shape[0]);
        const auto seq_len = std::max<size_t>(1, index_query_shape[1]);
        const auto h_iq = index_heads_4d(params.get_input_layout(QSAIn::INDEX_QUERY));
        const auto tokens = batch * seq_len;

        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(tokens * h_iq * desc->indexer_head_dim, ov::element::f16);
        bufs.emplace_back(score_scratch_elems(params, *desc), ov::element::f32);
        return bufs;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        kernel_dump_info.clear_entries();
        update_rt_params(instance);

        const auto& params = *instance.get_impl_params();
        auto* rtp = &rt();
        auto prepare_event = execute_stage(copy_state(events, instance), instance, prepare);
        if (is_decode_shape(params)) {
            // Chunks reuse one bounded BLOCK_SCORES buffer, so they run strictly in sequence;
            // the stages are re-dispatched per chunk, hence the explicit refresh.
            event::ptr last = prepare_event;
            for (size_t begin = 0; begin < rtp->num_tokens; begin += rtp->chunk_rows) {
                rtp->chunk_row_begin = begin;
                rtp->chunk_row_end = std::min(rtp->num_tokens, begin + rtp->chunk_rows);
                force_stage_refresh(*score_partition);
                force_stage_refresh(*topk_decode);
                auto scored = execute_stage({last}, instance, score_partition);
                last = execute_stage({scored}, instance, topk_decode);
            }
            return last;
        }
        rtp->chunk_row_begin = 0;
        rtp->chunk_row_end = rtp->num_tokens;
        if (use_dpas_score(params)) {
            auto scored = execute_stage({prepare_event}, instance, score_dpas);
            return execute_stage({scored}, instance, topk);
        }
        return execute_stage({prepare_event}, instance, score_topk_fused);
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        force_stages_refresh(*this);
        auto* rtp = &rt();

        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<qsa_indexer>();

        const auto& index_query_layout = params.get_input_layout(QSAIn::INDEX_QUERY);
        const auto& index_query_shape = index_query_layout.get_shape();
        const size_t batch = index_query_shape[0];
        const size_t seq_len = index_query_shape[1];

        rtp->batch = batch;
        rtp->seq_len = seq_len;
        rtp->num_tokens = batch * seq_len;
        rtp->index_query_heads = index_heads_4d(index_query_layout);
        rtp->index_key_heads = index_heads_4d(params.get_input_layout(QSAIn::RAW_INDEX_KEYS));
        rtp->summary_stride = params.get_input_layout(QSAIn::BLOCK_INDEX_KEYS_CACHE).get_shape()[2];

        // key_length stays in GPU memory: no host readback, so every size below is a bound from
        // shapes only and the kernels derive the real work (and drop the excess) from key_length.
        const auto r = desc->key_block_size;
        // blocks spanned by <= L consecutive tokens (L == 0 spans none)
        rtp->prepare_blocks_per_row = seq_len == 0 ? 0 : (seq_len - 1) / r + 2;
        rtp->prepare_wg_count = batch * rtp->prepare_blocks_per_row;
        const size_t max_blocks = std::max<size_t>(1, rtp->summary_stride);
        rtp->max_blocks = max_blocks;
        rtp->num_partitions = std::max<size_t>(1, (max_blocks + score_partition_blocks - 1) / score_partition_blocks);
        rtp->chunk_rows = std::max<size_t>(1, std::min(rtp->num_tokens, score_scratch_elems(params, *desc) / max_blocks));
        rtp->chunk_row_begin = 0;
        rtp->chunk_row_end = rtp->num_tokens;

        rtp->score_tile_count = batch * ((seq_len + dpas_query_tile - 1) / dpas_query_tile);
    }

private:
    /// Functional state: every state output starts as a copy of its input, then prepare
    /// overwrites the entries this call produces. Inputs and outputs are distinct buffers,
    /// so prepare reads the old incomplete block without a read-before-write hazard.
    std::vector<event::ptr> copy_state(const std::vector<event::ptr>& events, primitive_inst& instance) {
        auto& stream = instance.get_network().get_stream();
        if (stream.get_queue_type() == QueueTypes::out_of_order) {
            stream.wait_for_events(events);
        }
        std::vector<event::ptr> deps = events;
        static constexpr std::array<std::pair<size_t, size_t>, 3> state = {std::pair{QSAIn::BLOCK_INDEX_KEYS_CACHE, QSAOut::BLOCK_INDEX_KEYS_CACHE},
                                                                           std::pair{QSAIn::PAST_INCOMPLETE_RAW_K, QSAOut::PAST_INCOMPLETE_RAW_K},
                                                                           std::pair{QSAIn::PAST_INCOMPLETE_ROPE, QSAOut::PAST_INCOMPLETE_ROPE}};
        const auto& params = *instance.get_impl_params();
        for (const auto& [in, out] : state) {
            auto src = instance.input_memory_ptr(in);
            auto dst = instance.output_memory_ptr(out);
            const auto bytes = params.get_output_layout(out).bytes_count();
            OPENVINO_ASSERT(!params.get_input_layout(in).data_padding && !params.get_output_layout(out).data_padding,
                            "[GPU][CM] qsa_indexer: padded state buffers are not supported");
            OPENVINO_ASSERT(src->size() >= bytes && dst->size() >= bytes, "[GPU][CM] qsa_indexer: state buffer is too small");
            const bool aliased = src == dst || (src->buffer_ptr() != nullptr && src->buffer_ptr() == dst->buffer_ptr());
            // An aliased summary is safe (prepare only appends to it); the incomplete-block
            // state is read and rewritten by different work-groups of one dispatch.
            OPENVINO_ASSERT(!aliased || in == QSAIn::BLOCK_INDEX_KEYS_CACHE, "[GPU][CM] qsa_indexer: incomplete-block state input and output must not alias");
            if (bytes == 0 || aliased) {
                continue;
            }
            deps.push_back(dst->copy_from(stream, *src, 0, 0, bytes, false));
        }
        return deps;
    }
};

}  // namespace

bool QSAIndexerCMImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::qsa_indexer>());

    if (node.has_fused_primitives()) {
        return false;
    }

    auto& engine = node.get_program().get_engine();
    const auto& config = node.get_program().get_config();
    const auto& info = engine.get_device_info();
    if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
        return false;
    }

    const auto desc = node.as<cldnn::qsa_indexer>().get_primitive();
    std::string reason;
    if (!ov::intel_gpu::op::QSAIndexer::validate_config(desc->get_config(), reason)) {
        return false;
    }

    if (desc->num_selection_heads != 1 || desc->indexer_head_dim % 16 != 0) {
        return false;
    }
    // B and L may be dynamic; only the head axes are needed statically.
    const auto& index_query_ps = node.get_input_layout(QSAIn::INDEX_QUERY).get_partial_shape();
    const auto& raw_index_keys_ps = node.get_input_layout(QSAIn::RAW_INDEX_KEYS).get_partial_shape();
    if (index_query_ps.rank().is_dynamic() || index_query_ps.size() != 4 || index_query_ps[2].is_dynamic()) {
        return false;
    }
    if (raw_index_keys_ps.rank().is_dynamic() || raw_index_keys_ps.size() != 4 || raw_index_keys_ps[2].is_dynamic() || raw_index_keys_ps[2].get_length() != 1) {
        return false;
    }

    if (node.get_input_layout(QSAIn::INDEX_QUERY).data_type != ov::element::f16 || node.get_input_layout(QSAIn::RAW_INDEX_KEYS).data_type != ov::element::f16) {
        return false;
    }
    return true;
}

std::unique_ptr<cldnn::primitive_impl> QSAIndexerCMImplementationManager::create_impl(const cldnn::program_node& node,
                                                                                      const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::qsa_indexer>());
    return std::make_unique<QSAIndexerCmImpl>(params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::qsa_indexer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::QSAIndexerCmImpl)

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../primitive_cm_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_qsa_indexer.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::cm {

using namespace cldnn;
using QSAIn = ov::intel_gpu::op::PagedQSAIndexer::InputIdx;

namespace {

PagedQSAIndexerRuntimeParams* rt_of(ImplRuntimeParams* rt) {
    auto* p = static_cast<PagedQSAIndexerRuntimeParams*>(rt);
    OPENVINO_ASSERT(p != nullptr, "[GPU][CM] paged_qsa_indexer runtime params are not initialized");
    return p;
}

size_t index_heads(const layout& l, size_t indexer_head_dim) {
    const auto& ps = l.get_partial_shape();
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() >= 2 && ps[1].is_static() && indexer_head_dim != 0,
                    "[GPU][CM] paged_qsa_indexer: cannot derive index head count");
    return static_cast<size_t>(ps[1].get_length()) / indexer_head_dim;
}

JitConstants make_jit(const std::shared_ptr<const paged_qsa_indexer>& desc, const RuntimeParams& params) {
    const auto h_iq = index_heads(params.get_input_layout(QSAIn::Q_IDX), desc->indexer_head_dim);

    JitConstants jit;
    jit.make("QSA_IDX_QUERY_HEADS", h_iq);
    jit.make("QSA_IDX_HEAD_DIM", desc->indexer_head_dim);
    jit.make("QSA_IDX_ROTARY_DIM", desc->indexer_rotary_dim);
    jit.make("QSA_COMPRESS_RATIO", desc->compress_ratio);
    jit.make("QSA_PAGE_SIZE", desc->page_size);
    jit.make("QSA_BLOCK_TOPK", desc->block_topk);
    jit.make("QSA_ADDRESS_MODE", 0);
    jit.make("QSA_PREPARE_WG", h_iq + 1);
    jit.add(make_jit_constant("QSA_IDX_NORM_EPS", desc->eps));
    jit.add(make_jit_constant("QSA_IDX_SCALE", ov::op::internal::PagedQSAIndexer::effective_scale(desc->get_config())));
    return jit;
}

void push_u32(KernelData& kd, size_t value) {
    scalar_desc s;
    s.t = scalar_desc::Types::UINT32;
    s.v.u32 = static_cast<uint32_t>(value);
    kd.params.scalars.push_back(s);
}

constexpr auto get_qsa_indexer_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
}

// qsa_indexer_topk's cooperative work-group size (see qsa_topk_radix_wg.cm's proven choice).

constexpr size_t dpas_query_tile = 16;  // dpas N axis == GRF width on Xe2
constexpr size_t dpas_block_tile = 8;   // dpas RepeatCount (M axis)
constexpr size_t dpas_k_step = 16;      // fp16 K elements per dpas
constexpr size_t max_slm_bytes = 64 * 1024;
constexpr size_t score_wg_size = 16;  // threads per query tile in the dpas Q2 score kernel

// Decode-only Q2 score-partition kernel (see qsa_indexer.cpp's identical constant /
// rationale and measurements).
constexpr size_t score_partition_blocks = 32;

/// Upper bound on the decode BLOCK_SCORES scratch (old qsa.cpp's score_budget_bytes()).
/// QSA_TEST_SCORE_BUDGET_BYTES lets tests force the multi-chunk path with small shapes.
size_t score_budget_bytes() {
    if (const char* override_str = std::getenv("QSA_TEST_SCORE_BUDGET_BYTES")) {
        const auto value = std::strtoull(override_str, nullptr, 10);
        if (value > 0) {
            return static_cast<size_t>(value);
        }
    }
    return 128 * 1024 * 1024;
}

/// Whether every sequence in this call contributes exactly one query token. A
/// shape-agnostic impl keeps both the decode and prefill stages and decides per call.
bool is_decode_shape(const kernel_impl_params& params) {
    const auto& q_layout = params.get_input_layout(QSAIn::Q_IDX);
    const auto& past_lens_layout = params.get_input_layout(QSAIn::PAST_LENS);
    if (q_layout.is_dynamic() || past_lens_layout.is_dynamic()) {
        return false;
    }
    const auto tokens = q_layout.get_shape()[0];
    const auto seqs = past_lens_layout.get_shape()[0];
    return seqs > 0 && tokens == seqs;
}

bool may_decode(const kernel_impl_params& params) {
    return params.is_dynamic() || is_decode_shape(params);
}

bool may_prefill(const kernel_impl_params& params) {
    return params.is_dynamic() || !is_decode_shape(params);
}

/// Cooperative top-k work-group size: decode rows span the whole history, while prefill
/// rows are short and numerous, so smaller work-groups pack the GPU better.
constexpr size_t topk_wg_size(bool decode) {
    return decode ? 16 : 8;
}

/// f32 element capacity of BLOCK_SCORES. The row stride is bounded by the whole page pool
/// (not the actual context), so decode with many sequences and a large pool is what the
/// budget caps. Static shapes only, so allocation and per-call chunk planning agree.
size_t score_scratch_elems(const kernel_impl_params& params, const paged_qsa_indexer& desc) {
    const auto tokens = std::max<size_t>(1, params.get_input_layout(QSAIn::Q_IDX).get_shape()[0]);
    const auto& kcache_shape = params.get_input_layout(QSAIn::INDEXER_RAW_K_CACHE).get_shape();
    const auto max_blocks = std::max<size_t>(1, kcache_shape[0] * desc.page_size / desc.compress_ratio);
    if (!is_decode_shape(params)) {
        return tokens * max_blocks;
    }
    const auto budget_rows = std::max<size_t>(1, score_budget_bytes() / (max_blocks * sizeof(float)));
    return std::min(tokens, budget_rows) * max_blocks;
}

/// Whether the systolic qsa_indexer_score_dpas kernel can be used for this configuration.
/// Mirrors qsa.cpp's can_use_dpas_score(): a compile-time property of the descriptor plus
/// the GRF width, stable across iterations. Falls back to the scalar qsa_indexer_score
/// kernel when it does not hold.
bool can_use_dpas_score(const paged_qsa_indexer& desc, size_t h_iq, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;  // REG_N is 8 on Xe1, the kernel assumes a 16-wide N axis
    }
    if (desc.indexer_head_dim % dpas_k_step != 0) {
        return false;
    }
    if (desc.indexer_head_dim * sizeof(ov::float16) < 64) {
        return false;  // 2D block load surface width minimum
    }
    const auto summaries_per_page = desc.page_size / desc.compress_ratio;
    if (summaries_per_page == 0 || (summaries_per_page % dpas_block_tile != 0 && dpas_block_tile % summaries_per_page != 0)) {
        return false;  // the A tile must be whole pages, or a whole tile must fit in a page
    }
    const auto staged_bytes = h_iq * (desc.indexer_head_dim / dpas_k_step) * dpas_k_step * dpas_query_tile * sizeof(ov::float16);
    return staged_bytes <= max_slm_bytes;
}

// ===========================================================================
// Q1 - indexer prepare (paged addressing)
// ===========================================================================
class PagedPrepareGenerator : public KernelGenerator {
public:
    PagedPrepareGenerator() : KernelGenerator("qsa_indexer_prepare_paged", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<paged_qsa_indexer>(), params));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::Q_IDX});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::K_IDX});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::Q_NORM_WEIGHT});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::K_NORM_WEIGHT});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::ROPE_COS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::ROPE_SIN});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_RAW_K_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_ROPE_CACHE});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});
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
            push_u32(kd, rtp->num_seqs);
        }};
    }
};

// ===========================================================================
// Q2 prefill, systolic variant - score (paged addressing) on cm_dpas
// ===========================================================================
class PagedScoreDpasGenerator : public KernelGenerator {
public:
    PagedScoreDpasGenerator() : KernelGenerator("qsa_indexer_score_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<paged_qsa_indexer>(), params));
        jit.make("QSA_SCORE_WG", score_wg_size);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::BLOCK_SCORES});
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
            // One work-group owns one 16-query tile.
            wgs.global = {std::max<size_t>(1, rtp->score_tile_count) * score_wg_size, 1, 1};
            wgs.local = {score_wg_size, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->score_tile_count);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, rtp->max_blocks);
        }};
    }
};

// ===========================================================================
// Q2 - top-k finalize (paged addressing); writes directly to the op's own outputs
// ===========================================================================
class PagedTopkGenerator : public KernelGenerator {
public:
    explicit PagedTopkGenerator(bool decode = false)
        : KernelGenerator("qsa_indexer_topk", decode ? "_decode_cm" : "_cm"),
          m_decode(decode) {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<paged_qsa_indexer>(), params));
        jit.make("QSA_TOPK_WG", topk_wg_size(m_decode));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::BLOCK_SCORES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
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
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, rtp->max_blocks);
            push_u32(kd, rtp->chunk_row_begin);
            push_u32(kd, rtp->chunk_row_end);
        }};
    }

private:
    bool m_decode;
};

// ===========================================================================
// Q2 prefill/mixed without dpas - fused score + top-k (paged addressing)
// ===========================================================================
class PagedScoreTopkFusedGenerator : public KernelGenerator {
public:
    PagedScoreTopkFusedGenerator() : KernelGenerator("qsa_indexer_score_topk_fused", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<paged_qsa_indexer>(), params));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::BLOCK_SCORES});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
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
// Q2 decode, score-partition variant (paged addressing)
// ===========================================================================
class PagedScorePartitionGenerator : public KernelGenerator {
public:
    PagedScorePartitionGenerator() : KernelGenerator("qsa_indexer_score_partition", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<paged_qsa_indexer>(), params));
        jit.make("QSA_SCORE_PARTITION_BLOCKS", score_partition_blocks);
        // Rows whose blocks are all selected are never read by qsa_indexer_topk.
        jit.make("QSA_DENSE_SCORE_BYPASS", 1);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::BLOCK_SCORES});
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
            auto* rtp = rt_of(rt);
            auto& wgs = kd.params.workGroups;
            wgs.global = {std::max<size_t>(1, rtp->chunk_row_end - rtp->chunk_row_begin), 1, rtp->num_partitions};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, rtp->max_blocks);
            push_u32(kd, rtp->num_partitions);
            push_u32(kd, rtp->chunk_row_begin);
            push_u32(kd, rtp->chunk_row_end);
        }};
    }
};

class PagedQSAIndexerCmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::PagedQSAIndexerCmImpl)

    Stage::Ptr prepare = make_stage<PagedPrepareGenerator>();
    Stage::Ptr score_topk_fused = make_stage<PagedScoreTopkFusedGenerator>();
    Stage::Ptr score_dpas = make_stage<PagedScoreDpasGenerator>();
    Stage::Ptr score_partition = make_stage<PagedScorePartitionGenerator>();
    Stage::Ptr topk = make_stage<PagedTopkGenerator>();
    Stage::Ptr topk_decode = make_stage<PagedTopkGenerator>(true);

    PagedQSAIndexerCmImpl() : PrimitiveImplCM(PagedQSAIndexerCMImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<PagedQSAIndexerRuntimeParams>();
    }

    PagedQSAIndexerRuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<PagedQSAIndexerRuntimeParams>();
        }
        return *static_cast<PagedQSAIndexerRuntimeParams*>(m_rt_params.get());
    }

    // Q2: decode -> score_partition + topk (row-chunked); prefill/mixed -> score_dpas + topk,
    // or the fused score+top-k kernel when dpas is unavailable (old qsa.cpp's routing).
    explicit PagedQSAIndexerCmImpl(const RuntimeParams& params) : PagedQSAIndexerCmImpl() {
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
        return make_deep_copy<PagedQSAIndexerCmImpl>(this);
    }

    [[nodiscard]] bool requires_update(primitive_inst& /*inst*/, const kernel_impl_params& /*impl_params*/) const override {
        return true;
    }

    // Recomputed rather than cached: make_deep_copy() rebuilds the impl with the default
    // constructor and copies only _order and the per-stage kernel data.
    static bool use_dpas_score(const kernel_impl_params& params) {
        const auto desc = params.typed_desc<paged_qsa_indexer>();
        const auto h_iq = index_heads(params.get_input_layout(QSAIn::Q_IDX), desc->indexer_head_dim);
        return can_use_dpas_score(*desc, h_iq, params.get_device_info());
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        if (params.is_dynamic()) {
            return {};
        }
        const auto desc = params.typed_desc<paged_qsa_indexer>();
        const auto tokens = std::max<size_t>(1, params.get_input_layout(QSAIn::Q_IDX).get_shape()[0]);
        const auto h_iq = index_heads(params.get_input_layout(QSAIn::Q_IDX), desc->indexer_head_dim);

        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(tokens * h_iq * desc->indexer_head_dim, ov::element::f16);  // INDEXER_Q
        bufs.emplace_back(score_scratch_elems(params, *desc), ov::element::f32);  // BLOCK_SCORES
        return bufs;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        kernel_dump_info.clear_entries();
        update_rt_params(instance);

        const auto& params = *instance.get_impl_params();
        auto* rtp = &rt();
        auto prepare_event = execute_stage(events, instance, prepare);
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
        const auto desc = params.typed_desc<paged_qsa_indexer>();

        // past_lens / subsequence_begins stay in GPU memory: every size here is an upper bound taken
        // from shapes, and the kernels resolve the real sequences (and drop the excess work).
        const auto& q_layout = params.get_input_layout(QSAIn::Q_IDX);
        const size_t num_seqs = params.get_input_layout(QSAIn::PAST_LENS).get_shape()[0];
        rtp->num_seqs = num_seqs;
        rtp->num_tokens = q_layout.get_shape()[0];
        rtp->index_query_heads = index_heads(q_layout, desc->indexer_head_dim);
        rtp->index_key_heads = index_heads(params.get_input_layout(QSAIn::K_IDX), desc->indexer_head_dim);

        // Sequence i touches at most ceil(q_i / r) + 1 blocks, so all sequences together at most
        // tokens / r + 2 * num_seqs; and at most tokens / 16 + num_seqs score tiles.
        const auto r = desc->compress_ratio;
        rtp->prepare_wg_count = rtp->num_tokens / r + 2 * num_seqs;
        rtp->score_tile_count = rtp->num_tokens / dpas_query_tile + num_seqs;

        // No sequence can span more blocks than the page table lists.
        const size_t used_pages = params.get_input_layout(QSAIn::BLOCK_INDICES).get_shape()[0];
        const size_t max_blocks = std::max<size_t>(1, used_pages * desc->page_size / r);
        rtp->max_blocks = max_blocks;
        rtp->num_partitions = std::max<size_t>(1, (max_blocks + score_partition_blocks - 1) / score_partition_blocks);
        // max_blocks <= the pool-wide stride the buffer was sized for, so a chunk can hold more
        // rows than the static budget alone would allow.
        rtp->chunk_rows = std::max<size_t>(1, std::min(rtp->num_tokens, score_scratch_elems(params, *desc) / max_blocks));
        rtp->chunk_row_begin = 0;
        rtp->chunk_row_end = rtp->num_tokens;
    }
};

}  // namespace

bool PagedQSAIndexerCMImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::paged_qsa_indexer>());

    if (node.has_fused_primitives()) {
        return false;
    }

    auto& engine = node.get_program().get_engine();
    const auto& config = node.get_program().get_config();
    const auto& info = engine.get_device_info();
    if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
        return false;
    }

    const auto desc = node.as<cldnn::paged_qsa_indexer>().get_primitive();
    std::string reason;
    if (!ov::intel_gpu::op::PagedQSAIndexer::validate_config(desc->get_config(), reason)) {
        return false;
    }

    // v1 capability gate: only a single index-key head (H_ik == 1) and a single
    // selection head (h_sel == 1) are supported by this scalar kernel set.
    if (desc->h_sel != 1 || desc->indexer_head_dim % 16 != 0) {
        return false;
    }
    // The token axis may be dynamic; only the flattened head axis is needed statically.
    const auto& q_idx_ps = node.get_input_layout(QSAIn::Q_IDX).get_partial_shape();
    const auto& k_idx_ps = node.get_input_layout(QSAIn::K_IDX).get_partial_shape();
    if (q_idx_ps.rank().is_dynamic() || q_idx_ps.size() < 2 || q_idx_ps[1].is_dynamic()) {
        return false;
    }
    if (k_idx_ps.rank().is_dynamic() || k_idx_ps.size() < 2 || k_idx_ps[1].is_dynamic() ||
        static_cast<size_t>(k_idx_ps[1].get_length()) != desc->indexer_head_dim) {
        return false;
    }

    if (node.get_input_layout(QSAIn::Q_IDX).data_type != ov::element::f16 || node.get_input_layout(QSAIn::K_IDX).data_type != ov::element::f16) {
        return false;
    }
    return true;
}

std::unique_ptr<cldnn::primitive_impl> PagedQSAIndexerCMImplementationManager::create_impl(const cldnn::program_node& node,
                                                                                            const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::paged_qsa_indexer>());
    return std::make_unique<PagedQSAIndexerCmImpl>(params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_qsa_indexer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::PagedQSAIndexerCmImpl)

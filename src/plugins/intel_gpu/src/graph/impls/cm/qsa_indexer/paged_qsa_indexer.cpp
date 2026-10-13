// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer.hpp"

#include <cassert>
#include <cmath>
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

std::vector<int32_t> read_i32_input(const primitive_inst& instance, size_t input_idx) {
    auto mem = instance.input_memory_ptr(input_idx);
    mem_lock<int32_t, mem_lock_type::read> lock(mem, instance.get_network().get_stream());
    return {lock.begin(), lock.end()};
}

PagedQSAIndexerRuntimeParams* rt_of(ImplRuntimeParams* rt) {
    auto* p = static_cast<PagedQSAIndexerRuntimeParams*>(rt);
    OPENVINO_ASSERT(p != nullptr, "[GPU][CM] paged_qsa_indexer runtime params are not initialized");
    return p;
}

size_t index_heads(const layout& l, size_t indexer_head_dim) {
    const auto& shape = l.get_shape();
    OPENVINO_ASSERT(shape.size() >= 2 && indexer_head_dim != 0, "[GPU][CM] paged_qsa_indexer: cannot derive index head count");
    return shape[1] / indexer_head_dim;
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
    jit.add(make_jit_constant("QSA_IDX_SCALE", 1.0f / std::sqrt(static_cast<float>(desc->indexer_head_dim))));
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
constexpr size_t topk_wg_size = 16;

constexpr size_t dpas_query_tile = 16;  // dpas N axis == GRF width on Xe2
constexpr size_t dpas_block_tile = 8;   // dpas RepeatCount (M axis)
constexpr size_t dpas_k_step = 16;      // fp16 K elements per dpas
constexpr size_t max_slm_bytes = 64 * 1024;
constexpr size_t score_wg_size = 16;  // threads per query tile in the dpas Q2 score kernel

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
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::ROTARY_COS_SIN});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::POSITION_IDS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_RAW_K_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_POSITION_CACHE});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::PREPARE_MAP});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
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
        }};
    }
};

// ===========================================================================
// Q2 - score (paged addressing)
// ===========================================================================
class PagedScoreGenerator : public KernelGenerator {
public:
    PagedScoreGenerator() : KernelGenerator("qsa_indexer_score", "_cm") {}

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
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::SCORE_TILE_MAP});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::BLOCK_SCORES});
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
            // One work-group owns one 16-query tile.
            wgs.global = {std::max<size_t>(1, rtp->score_tile_count) * score_wg_size, 1, 1};
            wgs.local = {score_wg_size, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->score_tile_count);
            push_u32(kd, rtp->max_blocks);
        }};
    }
};

// ===========================================================================
// Q2 - top-k finalize (paged addressing); writes directly to the op's own outputs
// ===========================================================================
class PagedTopkGenerator : public KernelGenerator {
public:
    PagedTopkGenerator() : KernelGenerator("qsa_indexer_topk", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<paged_qsa_indexer>(), params));
        jit.make("QSA_TOPK_WG", topk_wg_size);
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
            // One work-group of topk_wg_size threads cooperatively finalizes one token's row.
            wgs.global = {std::max<size_t>(1, rtp->num_tokens) * topk_wg_size, 1, 1};
            wgs.local = {topk_wg_size, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, rtp->max_blocks);
        }};
    }
};

class PagedQSAIndexerCmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::PagedQSAIndexerCmImpl)

    Stage::Ptr prepare = make_stage<PagedPrepareGenerator>();
    Stage::Ptr score = make_stage<PagedScoreGenerator>();
    Stage::Ptr score_dpas = make_stage<PagedScoreDpasGenerator>();
    Stage::Ptr topk = make_stage<PagedTopkGenerator>();

    PagedQSAIndexerCmImpl() : PrimitiveImplCM(PagedQSAIndexerCMImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<PagedQSAIndexerRuntimeParams>();
    }

    PagedQSAIndexerRuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<PagedQSAIndexerRuntimeParams>();
        }
        return *static_cast<PagedQSAIndexerRuntimeParams*>(m_rt_params.get());
    }

    explicit PagedQSAIndexerCmImpl(const RuntimeParams& params) : PagedQSAIndexerCmImpl() {
        add_stage(prepare, params);
        add_stage(score, params);
        if (use_dpas_score(params)) {
            add_stage(score_dpas, params);
        }
        add_stage(topk, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedQSAIndexerCmImpl>(this);
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        update_rt_params(inst);
        PrimitiveImplCM::update(inst, impl_params);
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
        const auto desc = params.typed_desc<paged_qsa_indexer>();
        const auto tokens = std::max<size_t>(1, params.get_input_layout(QSAIn::Q_IDX).get_shape()[0]);
        const auto seqs = std::max<size_t>(1, params.get_input_layout(QSAIn::PAST_LENS).get_shape()[0]);
        const auto h_iq = index_heads(params.get_input_layout(QSAIn::Q_IDX), desc->indexer_head_dim);

        const auto& kcache_shape = params.get_input_layout(QSAIn::INDEXER_RAW_K_CACHE).get_shape();
        const auto max_blocks = std::max<size_t>(1, kcache_shape[0] * desc->page_size / desc->compress_ratio);
        const auto prepare_pairs = tokens + seqs;
        // One 16-query tile per sequence-local boundary, upper-bounded: at most
        // ceil(tokens/16) + seqs tiles (see qsa.cpp's identical score_tiles bound).
        const auto score_tiles = tokens / dpas_query_tile + seqs + 1;

        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(prepare_pairs * 2, ov::element::i32, true, false);  // PREPARE_MAP (host-filled)
        bufs.emplace_back(tokens * h_iq * desc->indexer_head_dim, ov::element::f16);  // INDEXER_Q
        bufs.emplace_back(tokens * max_blocks, ov::element::f32);  // BLOCK_SCORES
        bufs.emplace_back(score_tiles * 2, ov::element::i32, true, false);  // SCORE_TILE_MAP (host-filled)
        return bufs;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        update_rt_params(instance);
        upload_prepare_map(instance);

        const auto& params = *instance.get_impl_params();
        auto prepare_event = execute_stage(events, instance, prepare);
        auto score_event = use_dpas_score(params) ? execute_stage({prepare_event}, instance, score_dpas)
                                                   : execute_stage({prepare_event}, instance, score);
        auto topk_event = execute_stage({score_event}, instance, topk);
        return topk_event;
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        auto* rtp = &rt();

        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<paged_qsa_indexer>();
        const auto past_lens = read_i32_input(instance, QSAIn::PAST_LENS);
        const auto subsequence_begins = read_i32_input(instance, QSAIn::SUBSEQUENCE_BEGINS);

        const size_t num_seqs = past_lens.size();
        OPENVINO_ASSERT(subsequence_begins.size() == num_seqs + 1, "[GPU][CM] paged_qsa_indexer: subsequence_begins must have num_seqs + 1 entries");

        rtp->num_seqs = num_seqs;
        rtp->num_tokens = num_seqs == 0 ? 0 : static_cast<size_t>(subsequence_begins.back());
        rtp->index_query_heads = index_heads(params.get_input_layout(QSAIn::Q_IDX), desc->indexer_head_dim);
        rtp->index_key_heads = index_heads(params.get_input_layout(QSAIn::K_IDX), desc->indexer_head_dim);

        const auto r = desc->compress_ratio;
        size_t max_blocks = 1;
        rtp->prepare_map.clear();
        for (size_t s = 0; s < num_seqs; s++) {
            const auto q_len = static_cast<size_t>(std::max(0, subsequence_begins[s + 1] - subsequence_begins[s]));
            const auto past_len = static_cast<size_t>(std::max(0, past_lens[s]));
            if (q_len == 0) {
                continue;
            }
            const size_t total_len = past_len + q_len;
            max_blocks = std::max(max_blocks, total_len / r);

            const size_t first_block = past_len / r;
            const size_t last_block = (total_len - 1) / r;
            for (size_t b = first_block; b <= last_block; b++) {
                rtp->prepare_map.push_back(static_cast<int32_t>(s));
                rtp->prepare_map.push_back(static_cast<int32_t>(b));
            }
        }
        rtp->prepare_wg_count = rtp->prepare_map.size() / 2;
        rtp->max_blocks = max_blocks;

        // Score tile map (dpas score only; harmless to build unconditionally - it is tiny
        // relative to prepare_map and uploaded only when the dpas stage is dispatched).
        rtp->score_tile_map.clear();
        for (size_t s = 0; s < num_seqs; s++) {
            const auto seq_begin = static_cast<size_t>(subsequence_begins[s]);
            const auto seq_end = static_cast<size_t>(subsequence_begins[s + 1]);
            for (size_t t = seq_begin; t < seq_end; t += dpas_query_tile) {
                rtp->score_tile_map.push_back(static_cast<int32_t>(s));
                rtp->score_tile_map.push_back(static_cast<int32_t>(t));
            }
        }
        rtp->score_tile_count = rtp->score_tile_map.size() / 2;
    }

private:
    void upload_prepare_map(primitive_inst& instance) {
        auto* rtp = &rt();
        if (!rtp->prepare_map.empty()) {
            auto mem = instance.get_intermediates_memories()[PagedQSAIndexerInternalBufIdx::PREPARE_MAP];
            OPENVINO_ASSERT(mem->count() >= rtp->prepare_map.size(), "[GPU][CM] paged_qsa_indexer prepare map buffer is too small");
            mem_lock<int32_t, mem_lock_type::write> lock(mem, instance.get_network().get_stream());
            std::copy(rtp->prepare_map.begin(), rtp->prepare_map.end(), lock.begin());
        }
        if (!rtp->score_tile_map.empty() && use_dpas_score(*instance.get_impl_params())) {
            auto mem = instance.get_intermediates_memories()[PagedQSAIndexerInternalBufIdx::SCORE_TILE_MAP];
            OPENVINO_ASSERT(mem->count() >= rtp->score_tile_map.size(), "[GPU][CM] paged_qsa_indexer score tile map buffer is too small");
            mem_lock<int32_t, mem_lock_type::write> lock(mem, instance.get_network().get_stream());
            std::copy(rtp->score_tile_map.begin(), rtp->score_tile_map.end(), lock.begin());
        }
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
    if (desc->h_sel != 1) {
        return false;
    }
    const auto h_ik = index_heads(node.get_input_layout(QSAIn::K_IDX), desc->indexer_head_dim);
    if (h_ik != 1) {
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

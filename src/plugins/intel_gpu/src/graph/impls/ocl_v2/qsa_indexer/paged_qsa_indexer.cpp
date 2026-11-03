// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "../primitive_ocl_base.hpp"
#include "../qsa_host_common.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_qsa_indexer.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::ocl {

using namespace cldnn;
using QSAIn = ov::intel_gpu::op::PagedQSAIndexer::InputIdx;

namespace {

/// Tokens per physical page: dim 2 of index_key_cache (there is no page_size attribute).
size_t page_size_of(const kernel_impl_params& params) {
    const auto& ps = params.get_input_layout(QSAIn::INDEX_KEY_CACHE).get_partial_shape();
    OPENVINO_ASSERT(ps.size() == 4 && ps[2].is_static(), "[GPU][OCL] paged_qsa_indexer: index_key_cache must have a static page_size dim");
    return static_cast<size_t>(ps[2].get_length());
}

PagedQSAIndexerRuntimeParams* rt_of(ImplRuntimeParams* rt) {
    auto* p = static_cast<PagedQSAIndexerRuntimeParams*>(rt);
    OPENVINO_ASSERT(p != nullptr, "[GPU][OCL] paged_qsa_indexer runtime params are not initialized");
    return p;
}

size_t index_heads(const layout& l, size_t indexer_head_dim) {
    const auto& ps = l.get_partial_shape();
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() >= 2 && ps[1].is_static() && indexer_head_dim != 0,
                    "[GPU][OCL] paged_qsa_indexer: cannot derive index head count");
    return static_cast<size_t>(ps[1].get_length()) / indexer_head_dim;
}

JitConstants make_jit(JitConstants jit, const std::shared_ptr<const paged_qsa_indexer>& desc, const RuntimeParams& params) {
    const auto h_iq = index_heads(params.get_input_layout(QSAIn::INDEX_QUERY), desc->indexer_head_dim);
    jit.make("QSA_IDX_QUERY_HEADS", h_iq);
    jit.make("QSA_IDX_HEAD_DIM", desc->indexer_head_dim);
    jit.make("QSA_IDX_ROTARY_DIM", desc->indexer_rotary_dim);
    jit.make("QSA_COMPRESS_RATIO", desc->key_block_size);
    jit.make("QSA_PAGE_SIZE", page_size_of(params));
    jit.make("QSA_BLOCK_TOPK", desc->block_topk);
    jit.make("QSA_ADDRESS_MODE", 0);
    jit.add(qsa::float_constant("QSA_IDX_NORM_EPS", desc->epsilon));
    jit.add(qsa::float_constant("QSA_IDX_SCALE", ov::op::internal::PagedQSAIndexer::effective_scale(desc->get_config())));
    return jit;
}

constexpr size_t max_slm_bytes = 64 * 1024;
constexpr size_t score_wg_size = 16;  // subgroups per query tile in the systolic score kernel
constexpr size_t score_partition_blocks = 32;

bool is_decode_shape(const kernel_impl_params& params) {
    const auto& q_layout = params.get_input_layout(QSAIn::INDEX_QUERY);
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

/// Whether the systolic qsa_indexer_score_dpas kernel can be used (a property of the descriptor and the device).
bool can_use_dpas_score(const paged_qsa_indexer& desc, size_t page_size, size_t h_iq, const cldnn::device_info& info) {
    if (!qsa::xmx_allowed(info)) {
        return false;
    }
    if (desc.indexer_head_dim % qsa::dpas_k_step != 0 || desc.indexer_head_dim * sizeof(ov::float16) < 64) {
        return false;
    }
    const auto summaries_per_page = page_size / desc.key_block_size;
    if (summaries_per_page == 0 || (summaries_per_page % 8 != 0 && 8 % summaries_per_page != 0)) {
        return false;
    }
    const auto staged_bytes = h_iq * (desc.indexer_head_dim / qsa::dpas_k_step) * qsa::dpas_k_step * qsa::dpas_query_tile * sizeof(ov::float16);
    return staged_bytes <= max_slm_bytes;
}

bool use_dpas_score(const kernel_impl_params& params) {
    const auto desc = params.typed_desc<paged_qsa_indexer>();
    const auto h_iq = index_heads(params.get_input_layout(QSAIn::INDEX_QUERY), desc->indexer_head_dim);
    return can_use_dpas_score(*desc, page_size_of(params), h_iq, params.get_device_info());
}

/// Row pitch the scorer / finalizer agree on when the pool holds the most blocks it can ever list.
size_t pool_pitch(const kernel_impl_params& params, const paged_qsa_indexer& desc) {
    const auto& kcache_shape = params.get_input_layout(QSAIn::INDEX_KEY_CACHE).get_shape();
    return qsa::score_pitch(kcache_shape[0] * page_size_of(params) / desc.key_block_size);
}

/// f32 element capacity of BLOCK_SCORES. Only the systolic prefill scorer writes one row per token; every other path
/// is row-chunked under the score budget. Static shapes only, so allocation and per-call planning agree.
size_t score_scratch_elems(const kernel_impl_params& params, const paged_qsa_indexer& desc) {
    const auto tokens = std::max<size_t>(1, params.get_input_layout(QSAIn::INDEX_QUERY).get_shape()[0]);
    const auto pitch = pool_pitch(params, desc);
    if (!is_decode_shape(params) && use_dpas_score(params)) {
        return tokens * pitch;
    }
    const auto budget_rows = std::max<size_t>(1, qsa::score_budget_bytes() / (pitch * sizeof(float)));
    return std::min(tokens, budget_rows) * pitch;
}

class PagedPrepareGenerator : public qsa::QsaGenerator {
public:
    PagedPrepareGenerator() : QsaGenerator("qsa_indexer_prepare_paged", "qsa_indexer_prepare_paged") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        return make_jit(QsaGenerator::get_jit_constants(params), params.typed_desc<paged_qsa_indexer>(), params);
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        for (auto idx : {QSAIn::INDEX_QUERY,
                         QSAIn::RAW_INDEX_KEYS,
                         QSAIn::Q_NORM_WEIGHT,
                         QSAIn::K_NORM_WEIGHT,
                         QSAIn::ROPE_COS,
                         QSAIn::ROPE_SIN,
                         QSAIn::PAST_LENS,
                         QSAIn::SUBSEQUENCE_BEGINS,
                         QSAIn::BLOCK_INDICES,
                         QSAIn::BLOCK_INDICES_BEGINS,
                         QSAIn::INDEX_KEY_CACHE,
                         QSAIn::BLOCK_INDEX_KEYS_CACHE,
                         QSAIn::INDEX_BLOCK_ROPE_CACHE}) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(idx)});
        }
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
            const auto sg = qsa::subgroup_size(params.get_device_info());
            const size_t wg_count = std::max<size_t>(1, rtp->prepare_wg_count);
            auto& wgs = kd.params.workGroups;
            wgs.global = {sg, rtp->index_query_heads + 1, wg_count};
            wgs.local = {sg, 1, 1};

            kd.params.scalars.clear();
            qsa::push_u32(kd, rtp->prepare_wg_count);
            qsa::push_u32(kd, rtp->num_seqs);
        }};
    }
};

class PagedScoreDpasGenerator : public qsa::QsaGenerator {
public:
    PagedScoreDpasGenerator() : QsaGenerator("qsa_indexer_score_dpas", "qsa_indexer_score_dpas", true) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_jit(QsaGenerator::get_jit_constants(params), params.typed_desc<paged_qsa_indexer>(), params);
        jit.make("QSA_SCORE_WG", score_wg_size);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::INDEXER_Q});
        for (auto idx : {QSAIn::BLOCK_INDEX_KEYS_CACHE, QSAIn::PAST_LENS, QSAIn::SUBSEQUENCE_BEGINS, QSAIn::BLOCK_INDICES, QSAIn::BLOCK_INDICES_BEGINS}) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(idx)});
        }
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
            const auto sg = qsa::subgroup_size(params.get_device_info());
            auto& wgs = kd.params.workGroups;
            // One work-group owns one 16-query tile.
            wgs.global = {std::max<size_t>(1, rtp->score_tile_count) * score_wg_size * sg, 1, 1};
            wgs.local = {score_wg_size * sg, 1, 1};

            kd.params.scalars.clear();
            qsa::push_u32(kd, rtp->score_tile_count);
            qsa::push_u32(kd, rtp->num_seqs);
            qsa::push_u32(kd, rtp->pitch);
        }};
    }
};

class PagedTopkGenerator : public qsa::QsaGenerator {
public:
    explicit PagedTopkGenerator(bool decode = false)
        : QsaGenerator("qsa_indexer_topk", decode ? "qsa_indexer_topk_decode" : "qsa_indexer_topk"),
          m_decode(decode) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_jit(QsaGenerator::get_jit_constants(params), params.typed_desc<paged_qsa_indexer>(), params);
        jit.make("QSA_TOPK_WG", qsa::topk_max_subgroups);
        if (m_decode) {
            jit.make("QSA_TOPK_FIXED_WG", 1);
        }
        jit.make("QSA_TOPK_REG_KEYS", qsa::topk_reg_keys);
        jit.make("QSA_TOPK_THRESHOLDS", qsa::topk_thresholds);
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
            const auto sg = qsa::subgroup_size(params.get_device_info());
            // A decode row is one latency-bound work-group: use all subgroups.
            const auto wg = (decode ? qsa::topk_max_subgroups : qsa::topk_wg_subgroups(rtp->pitch)) * sg;
            auto& wgs = kd.params.workGroups;
            // One work-group cooperatively finalizes one token's row.
            wgs.global = {std::max<size_t>(1, rtp->chunk_row_end - rtp->chunk_row_begin) * wg, 1, 1};
            wgs.local = {wg, 1, 1};

            kd.params.scalars.clear();
            qsa::push_u32(kd, rtp->num_tokens);
            qsa::push_u32(kd, rtp->num_seqs);
            qsa::push_u32(kd, rtp->pitch);
            qsa::push_u32(kd, rtp->chunk_row_begin);
            qsa::push_u32(kd, rtp->chunk_row_end);
        }};
    }

private:
    bool m_decode;
};

class PagedScorePartitionGenerator : public qsa::QsaGenerator {
public:
    PagedScorePartitionGenerator() : QsaGenerator("qsa_indexer_score_partition", "qsa_indexer_score_partition") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_jit(QsaGenerator::get_jit_constants(params), params.typed_desc<paged_qsa_indexer>(), params);
        jit.make("QSA_SCORE_PARTITION_BLOCKS", score_partition_blocks);
        // Rows whose blocks are all selected are never read by qsa_indexer_topk.
        jit.make("QSA_DENSE_SCORE_BYPASS", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, PagedQSAIndexerInternalBufIdx::INDEXER_Q});
        for (auto idx : {QSAIn::BLOCK_INDEX_KEYS_CACHE, QSAIn::PAST_LENS, QSAIn::SUBSEQUENCE_BEGINS, QSAIn::BLOCK_INDICES, QSAIn::BLOCK_INDICES_BEGINS}) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(idx)});
        }
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
            const auto sg = qsa::subgroup_size(params.get_device_info());
            auto& wgs = kd.params.workGroups;
            wgs.global = {std::max<size_t>(1, rtp->chunk_row_end - rtp->chunk_row_begin) * sg, 1, rtp->num_partitions};
            wgs.local = {sg, 1, 1};

            kd.params.scalars.clear();
            qsa::push_u32(kd, rtp->num_tokens);
            qsa::push_u32(kd, rtp->num_seqs);
            qsa::push_u32(kd, rtp->pitch);
            qsa::push_u32(kd, rtp->num_partitions);
            qsa::push_u32(kd, rtp->chunk_row_begin);
            qsa::push_u32(kd, rtp->chunk_row_end);
        }};
    }
};

class PagedQSAIndexerImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PagedQSAIndexerImpl)

    Stage::Ptr prepare = make_stage<PagedPrepareGenerator>();
    Stage::Ptr score_dpas = make_stage<PagedScoreDpasGenerator>();
    Stage::Ptr score_partition = make_stage<PagedScorePartitionGenerator>();
    Stage::Ptr topk = make_stage<PagedTopkGenerator>();
    Stage::Ptr topk_decode = make_stage<PagedTopkGenerator>(true);

    PagedQSAIndexerImpl() : PrimitiveImplOCL(PagedQSAIndexerImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<PagedQSAIndexerRuntimeParams>();
    }

    PagedQSAIndexerRuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<PagedQSAIndexerRuntimeParams>();
        }
        return *static_cast<PagedQSAIndexerRuntimeParams*>(m_rt_params.get());
    }

    // Decode: prepare -> [score_partition -> topk_decode] per row chunk. Prefill/mixed: score_dpas -> topk when the
    // systolic scorer is available, otherwise the same row-chunked partition scorer with the prefill top-k.
    explicit PagedQSAIndexerImpl(const RuntimeParams& params) : PagedQSAIndexerImpl() {
        const bool xmx = use_dpas_score(params);
        add_stage(prepare, params);
        if (may_decode(params) || (may_prefill(params) && !xmx)) {
            add_stage(score_partition, params);
        }
        if (may_decode(params)) {
            add_stage(topk_decode, params);
        }
        if (may_prefill(params)) {
            if (xmx) {
                add_stage(score_dpas, params);
            }
            add_stage(topk, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedQSAIndexerImpl>(this);
    }

    [[nodiscard]] bool requires_update(primitive_inst& /*inst*/, const kernel_impl_params& /*impl_params*/) const override {
        return true;
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        if (params.is_dynamic()) {
            return {};
        }
        const auto desc = params.typed_desc<paged_qsa_indexer>();
        const auto tokens = std::max<size_t>(1, params.get_input_layout(QSAIn::INDEX_QUERY).get_shape()[0]);
        const auto h_iq = index_heads(params.get_input_layout(QSAIn::INDEX_QUERY), desc->indexer_head_dim);

        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(tokens * h_iq * desc->indexer_head_dim, ov::element::f16);  // INDEXER_Q
        bufs.emplace_back(score_scratch_elems(params, *desc), ov::element::f32);      // BLOCK_SCORES
        return bufs;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        kernel_dump_info.clear_entries();
        update_rt_params(instance);

        const auto& params = *instance.get_impl_params();
        auto* rtp = &rt();
        auto last = execute_stage(events, instance, prepare);
        const bool decode = is_decode_shape(params);
        if (!decode && use_dpas_score(params)) {
            rtp->chunk_row_begin = 0;
            rtp->chunk_row_end = rtp->num_tokens;
            auto scored = execute_stage({last}, instance, score_dpas);
            return execute_stage({scored}, instance, topk);
        }
        // Chunks reuse one bounded BLOCK_SCORES buffer, so they run strictly in sequence; the stages are
        // re-dispatched per chunk, hence the explicit refresh.
        Stage& finalizer = decode ? *topk_decode : *topk;
        for (size_t begin = 0; begin < rtp->num_tokens; begin += rtp->chunk_rows) {
            rtp->chunk_row_begin = begin;
            rtp->chunk_row_end = std::min(rtp->num_tokens, begin + rtp->chunk_rows);
            qsa::force_stage_refresh(*score_partition);
            qsa::force_stage_refresh(finalizer);
            auto scored = execute_stage({last}, instance, score_partition);
            last = execute_stage({scored}, instance, finalizer);
        }
        return last;
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        qsa::force_stages_refresh(*this);
        auto* rtp = &rt();

        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<paged_qsa_indexer>();

        const auto& q_layout = params.get_input_layout(QSAIn::INDEX_QUERY);
        const size_t num_seqs = params.get_input_layout(QSAIn::PAST_LENS).get_shape()[0];
        rtp->num_seqs = num_seqs;
        rtp->num_tokens = q_layout.get_shape()[0];
        rtp->index_query_heads = index_heads(q_layout, desc->indexer_head_dim);

        // Sequence i touches at most ceil(q_i / r) + 1 blocks, so all sequences together at most
        // tokens / r + 2 * num_seqs; and at most tokens / 16 + num_seqs score tiles.
        const auto r = desc->key_block_size;
        rtp->prepare_wg_count = rtp->num_tokens / r + 2 * num_seqs;
        rtp->score_tile_count = rtp->num_tokens / qsa::dpas_query_tile + num_seqs;

        // No sequence can span more blocks than the page table lists.
        const size_t used_pages = params.get_input_layout(QSAIn::BLOCK_INDICES).get_shape()[0];
        const size_t max_blocks = std::max<size_t>(1, used_pages * page_size_of(params) / r);
        rtp->pitch = qsa::score_pitch(max_blocks);
        rtp->num_partitions = std::max<size_t>(1, (max_blocks + score_partition_blocks - 1) / score_partition_blocks);
        // The runtime pitch never exceeds the pool-wide pitch the buffer was sized for, so a chunk can hold more rows
        // than the static budget alone would allow.
        rtp->chunk_rows = std::max<size_t>(1, std::min(rtp->num_tokens, score_scratch_elems(params, *desc) / rtp->pitch));
        rtp->chunk_row_begin = 0;
        rtp->chunk_row_end = rtp->num_tokens;
    }
};

}  // namespace

bool PagedQSAIndexerImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::paged_qsa_indexer>());

    if (node.has_fused_primitives()) {
        return false;
    }

    const auto& info = node.get_program().get_engine().get_device_info();
    if (!qsa::subgroups_available(info)) {
        return false;
    }

    const auto desc = node.as<cldnn::paged_qsa_indexer>().get_primitive();
    std::string reason;
    if (!ov::intel_gpu::op::PagedQSAIndexer::validate_config(desc->get_config(), reason)) {
        return false;
    }

    // Single selection head; the head dim must tile the subgroup and keep its rows 16-byte aligned.
    if (desc->num_selection_heads != 1 || desc->indexer_head_dim % 16 != 0 || desc->indexer_head_dim % qsa::subgroup_size(info) != 0) {
        return false;
    }
    const auto& index_query_ps = node.get_input_layout(QSAIn::INDEX_QUERY).get_partial_shape();
    const auto& raw_index_keys_ps = node.get_input_layout(QSAIn::RAW_INDEX_KEYS).get_partial_shape();
    if (index_query_ps.rank().is_dynamic() || index_query_ps.size() < 2 || index_query_ps[1].is_dynamic()) {
        return false;
    }
    if (raw_index_keys_ps.rank().is_dynamic() || raw_index_keys_ps.size() < 2 || raw_index_keys_ps[1].is_dynamic() ||
        static_cast<size_t>(raw_index_keys_ps[1].get_length()) != desc->indexer_head_dim) {
        return false;
    }

    return node.get_input_layout(QSAIn::INDEX_QUERY).data_type == ov::element::f16 &&
           node.get_input_layout(QSAIn::RAW_INDEX_KEYS).data_type == ov::element::f16;
}

std::unique_ptr<cldnn::primitive_impl> PagedQSAIndexerImplementationManager::create_impl(const cldnn::program_node& node,
                                                                                         const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::paged_qsa_indexer>());
    return std::make_unique<PagedQSAIndexerImpl>(params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PagedQSAIndexerImpl)

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../primitive_ocl_base.hpp"
#include "../qsa_host_common.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/qsa_indexer.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::ocl {

using namespace cldnn;
using QSAIn = ov::intel_gpu::op::QSAIndexer::InputIdx;
using QSAOut = ov::intel_gpu::op::QSAIndexer::OutputIdx;

namespace {

QSAIndexerRuntimeParams* rt_of(ImplRuntimeParams* rt) {
    auto* p = static_cast<QSAIndexerRuntimeParams*>(rt);
    OPENVINO_ASSERT(p != nullptr, "[GPU][OCL] qsa_indexer runtime params are not initialized");
    return p;
}

/// H_iq / H_ik, derived from the 4D [B, L, H, D_idx] index_query/raw_index_keys layouts (B/L may be dynamic).
size_t index_heads_4d(const layout& l) {
    const auto& ps = l.get_partial_shape();
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() >= 3 && ps[2].is_static(),
                    "[GPU][OCL] qsa_indexer: expected a rank-4 [B, L, H, D_idx] input with a static head count");
    return static_cast<size_t>(ps[2].get_length());
}

JitConstants make_jit(JitConstants jit, const std::shared_ptr<const qsa_indexer>& desc, const RuntimeParams& params) {
    const auto h_iq = index_heads_4d(params.get_input_layout(QSAIn::INDEX_QUERY));
    jit.make("QSA_IDX_QUERY_HEADS", h_iq);
    jit.make("QSA_IDX_HEAD_DIM", desc->indexer_head_dim);
    jit.make("QSA_IDX_ROTARY_DIM", desc->indexer_rotary_dim);
    jit.make("QSA_COMPRESS_RATIO", desc->key_block_size);
    jit.make("QSA_BLOCK_TOPK", desc->block_topk);
    jit.make("QSA_ADDRESS_MODE", 1);
    jit.add(qsa::float_constant("QSA_IDX_NORM_EPS", desc->epsilon));
    jit.add(qsa::float_constant("QSA_IDX_SCALE", ov::intel_gpu::op::QSAIndexer::effective_scale(desc->get_config())));
    return jit;
}

constexpr size_t max_slm_bytes = 64 * 1024;
constexpr size_t score_wg_size = 16;
constexpr size_t score_partition_blocks = 32;

bool is_decode_shape(const kernel_impl_params& params) {
    const auto& index_query_layout = params.get_input_layout(QSAIn::INDEX_QUERY);
    if (index_query_layout.is_dynamic()) {
        return false;
    }
    const auto& shape = index_query_layout.get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "[GPU][OCL] qsa_indexer: expected a rank-4 [B, L, H, D_idx] input");
    return shape[1] == 1;
}

bool may_decode(const kernel_impl_params& params) {
    return params.is_dynamic() || is_decode_shape(params);
}

bool may_prefill(const kernel_impl_params& params) {
    return params.is_dynamic() || !is_decode_shape(params);
}

/// Dense addressing has no page table, so unlike the paged gate there is no page/tile alignment constraint.
bool can_use_dpas_score(const qsa_indexer& desc, size_t h_iq, const cldnn::device_info& info) {
    if (!qsa::xmx_allowed(info)) {
        return false;
    }
    if (desc.indexer_head_dim % qsa::dpas_k_step != 0 || desc.indexer_head_dim * sizeof(ov::float16) < 64) {
        return false;
    }
    const auto staged_bytes = h_iq * (desc.indexer_head_dim / qsa::dpas_k_step) * qsa::dpas_k_step * qsa::dpas_query_tile * sizeof(ov::float16);
    return staged_bytes <= max_slm_bytes;
}

bool use_dpas_score(const kernel_impl_params& params) {
    const auto desc = params.typed_desc<qsa_indexer>();
    const auto h_iq = index_heads_4d(params.get_input_layout(QSAIn::INDEX_QUERY));
    return can_use_dpas_score(*desc, h_iq, params.get_device_info());
}

size_t score_scratch_elems(const kernel_impl_params& params, const qsa_indexer& /*desc*/) {
    const auto& index_query_shape = params.get_input_layout(QSAIn::INDEX_QUERY).get_shape();
    const auto tokens = std::max<size_t>(1, index_query_shape[0] * index_query_shape[1]);
    const auto& summary_shape = params.get_input_layout(QSAIn::BLOCK_INDEX_KEYS_CACHE).get_shape();
    const auto pitch = qsa::score_pitch(summary_shape[2]);
    if (!is_decode_shape(params) && use_dpas_score(params)) {
        return tokens * pitch;
    }
    const auto budget_rows = std::max<size_t>(1, qsa::score_budget_bytes() / (pitch * sizeof(float)));
    return std::min(tokens, budget_rows) * pitch;
}

class DensePrepareGenerator : public qsa::QsaGenerator {
public:
    DensePrepareGenerator() : QsaGenerator("qsa_indexer_prepare_dense", "qsa_indexer_prepare_dense") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        return make_jit(QsaGenerator::get_jit_constants(params), params.typed_desc<qsa_indexer>(), params);
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        for (auto idx : {QSAIn::INDEX_QUERY,
                         QSAIn::RAW_INDEX_KEYS,
                         QSAIn::Q_NORM_WEIGHT,
                         QSAIn::K_NORM_WEIGHT,
                         QSAIn::ROPE_COS,
                         QSAIn::ROPE_SIN,
                         QSAIn::KEY_LENGTH,
                         QSAIn::PAST_INCOMPLETE_RAW_K,
                         QSAIn::PAST_INCOMPLETE_ROPE}) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(idx)});
        }
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
            const auto sg = qsa::subgroup_size(params.get_device_info());
            const size_t wg_count = std::max<size_t>(1, rtp->prepare_wg_count);
            auto& wgs = kd.params.workGroups;
            wgs.global = {sg, rtp->index_query_heads + 1, wg_count};
            wgs.local = {sg, 1, 1};

            kd.params.scalars.clear();
            qsa::push_u32(kd, rtp->prepare_wg_count);
            qsa::push_u32(kd, rtp->prepare_blocks_per_row);
            qsa::push_u32(kd, rtp->seq_len);
            qsa::push_u32(kd, rtp->summary_stride);
        }};
    }
};

class DenseScoreDpasGenerator : public qsa::QsaGenerator {
public:
    DenseScoreDpasGenerator() : QsaGenerator("qsa_indexer_score_dpas", "qsa_indexer_score_dpas", true) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_jit(QsaGenerator::get_jit_constants(params), params.typed_desc<qsa_indexer>(), params);
        jit.make("QSA_SCORE_WG", score_wg_size);
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
            const auto sg = qsa::subgroup_size(params.get_device_info());
            auto& wgs = kd.params.workGroups;
            wgs.global = {std::max<size_t>(1, rtp->score_tile_count) * score_wg_size * sg, 1, 1};
            wgs.local = {score_wg_size * sg, 1, 1};

            kd.params.scalars.clear();
            qsa::push_u32(kd, rtp->score_tile_count);
            qsa::push_u32(kd, rtp->seq_len);
            qsa::push_u32(kd, rtp->summary_stride);
            qsa::push_u32(kd, rtp->pitch);
        }};
    }
};

class DenseTopkGenerator : public qsa::QsaGenerator {
public:
    explicit DenseTopkGenerator(bool decode = false)
        : QsaGenerator("qsa_indexer_topk", decode ? "qsa_indexer_topk_decode" : "qsa_indexer_topk"),
          m_decode(decode) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_jit(QsaGenerator::get_jit_constants(params), params.typed_desc<qsa_indexer>(), params);
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
            const auto sg = qsa::subgroup_size(params.get_device_info());
            // A decode row is one latency-bound work-group: use all subgroups.
            const auto wg = (decode ? qsa::topk_max_subgroups : qsa::topk_wg_subgroups(rtp->pitch)) * sg;
            auto& wgs = kd.params.workGroups;
            wgs.global = {std::max<size_t>(1, rtp->chunk_row_end - rtp->chunk_row_begin) * wg, 1, 1};
            wgs.local = {wg, 1, 1};

            kd.params.scalars.clear();
            qsa::push_u32(kd, rtp->num_tokens);
            qsa::push_u32(kd, rtp->seq_len);
            qsa::push_u32(kd, rtp->pitch);
            qsa::push_u32(kd, rtp->chunk_row_begin);
            qsa::push_u32(kd, rtp->chunk_row_end);
        }};
    }

private:
    bool m_decode;
};

class DenseScorePartitionGenerator : public qsa::QsaGenerator {
public:
    DenseScorePartitionGenerator() : QsaGenerator("qsa_indexer_score_partition", "qsa_indexer_score_partition") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_jit(QsaGenerator::get_jit_constants(params), params.typed_desc<qsa_indexer>(), params);
        jit.make("QSA_SCORE_PARTITION_BLOCKS", score_partition_blocks);
        jit.make("QSA_DENSE_SCORE_BYPASS", 1);
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
            const auto sg = qsa::subgroup_size(params.get_device_info());
            auto& wgs = kd.params.workGroups;
            wgs.global = {std::max<size_t>(1, rtp->chunk_row_end - rtp->chunk_row_begin) * sg, 1, rtp->num_partitions};
            wgs.local = {sg, 1, 1};

            kd.params.scalars.clear();
            qsa::push_u32(kd, rtp->num_tokens);
            qsa::push_u32(kd, rtp->seq_len);
            qsa::push_u32(kd, rtp->summary_stride);
            qsa::push_u32(kd, rtp->pitch);
            qsa::push_u32(kd, rtp->num_partitions);
            qsa::push_u32(kd, rtp->chunk_row_begin);
            qsa::push_u32(kd, rtp->chunk_row_end);
        }};
    }
};

class QSAIndexerImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::QSAIndexerImpl)

    Stage::Ptr prepare = make_stage<DensePrepareGenerator>();
    Stage::Ptr score_dpas = make_stage<DenseScoreDpasGenerator>();
    Stage::Ptr score_partition = make_stage<DenseScorePartitionGenerator>();
    Stage::Ptr topk = make_stage<DenseTopkGenerator>();
    Stage::Ptr topk_decode = make_stage<DenseTopkGenerator>(true);

    QSAIndexerImpl() : PrimitiveImplOCL(QSAIndexerImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<QSAIndexerRuntimeParams>();
    }

    QSAIndexerRuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<QSAIndexerRuntimeParams>();
        }
        return *static_cast<QSAIndexerRuntimeParams*>(m_rt_params.get());
    }

    explicit QSAIndexerImpl(const RuntimeParams& params) : QSAIndexerImpl() {
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
        return make_deep_copy<QSAIndexerImpl>(this);
    }

    [[nodiscard]] bool requires_update(primitive_inst& /*inst*/, const kernel_impl_params& /*impl_params*/) const override {
        return true;
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

        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(batch * seq_len * h_iq * desc->indexer_head_dim, ov::element::f16);
        bufs.emplace_back(score_scratch_elems(params, *desc), ov::element::f32);
        return bufs;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        kernel_dump_info.clear_entries();
        update_rt_params(instance);

        const auto& params = *instance.get_impl_params();
        auto* rtp = &rt();
        auto last = execute_stage(copy_state(events, instance), instance, prepare);
        const bool decode = is_decode_shape(params);
        if (!decode && use_dpas_score(params)) {
            rtp->chunk_row_begin = 0;
            rtp->chunk_row_end = rtp->num_tokens;
            auto scored = execute_stage({last}, instance, score_dpas);
            return execute_stage({scored}, instance, topk);
        }
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
        const auto desc = params.typed_desc<qsa_indexer>();

        const auto& index_query_layout = params.get_input_layout(QSAIn::INDEX_QUERY);
        const auto& index_query_shape = index_query_layout.get_shape();
        const size_t batch = index_query_shape[0];
        const size_t seq_len = index_query_shape[1];

        rtp->batch = batch;
        rtp->seq_len = seq_len;
        rtp->num_tokens = batch * seq_len;
        rtp->index_query_heads = index_heads_4d(index_query_layout);
        rtp->summary_stride = params.get_input_layout(QSAIn::BLOCK_INDEX_KEYS_CACHE).get_shape()[2];

        // key_length stays in GPU memory: every size below is a bound from shapes only and the kernels derive the real
        // work (and drop the excess) from key_length.
        const auto r = desc->key_block_size;
        rtp->prepare_blocks_per_row = seq_len == 0 ? 0 : (seq_len - 1) / r + 2;
        rtp->prepare_wg_count = batch * rtp->prepare_blocks_per_row;
        const size_t max_blocks = std::max<size_t>(1, rtp->summary_stride);
        rtp->pitch = qsa::score_pitch(max_blocks);
        rtp->num_partitions = std::max<size_t>(1, (max_blocks + score_partition_blocks - 1) / score_partition_blocks);
        rtp->chunk_rows = std::max<size_t>(1, std::min(rtp->num_tokens, score_scratch_elems(params, *desc) / rtp->pitch));
        rtp->chunk_row_begin = 0;
        rtp->chunk_row_end = rtp->num_tokens;
        rtp->score_tile_count = batch * ((seq_len + qsa::dpas_query_tile - 1) / qsa::dpas_query_tile);
    }

private:
    /// Functional state: every state output starts as a copy of its input, then prepare overwrites the entries this call
    /// produces. Inputs and outputs are distinct buffers, so prepare reads the old incomplete block without a hazard.
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
                            "[GPU][OCL] qsa_indexer: padded state buffers are not supported");
            OPENVINO_ASSERT(src->size() >= bytes && dst->size() >= bytes, "[GPU][OCL] qsa_indexer: state buffer is too small");
            const bool aliased = src == dst || (src->buffer_ptr() != nullptr && src->buffer_ptr() == dst->buffer_ptr());
            // An aliased summary is safe (prepare only appends to it); the incomplete-block state is read and rewritten by
            // different work-groups of one dispatch.
            OPENVINO_ASSERT(!aliased || in == QSAIn::BLOCK_INDEX_KEYS_CACHE, "[GPU][OCL] qsa_indexer: incomplete-block state input and output must not alias");
            if (bytes == 0 || aliased) {
                continue;
            }
            deps.push_back(dst->copy_from(stream, *src, 0, 0, bytes, false));
        }
        return deps;
    }
};

}  // namespace

bool QSAIndexerImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::qsa_indexer>());

    if (!qsa::ocl_impl_allowed() || node.has_fused_primitives()) {
        return false;
    }

    const auto& info = node.get_program().get_engine().get_device_info();
    if (!qsa::subgroups_available(info)) {
        return false;
    }

    const auto desc = node.as<cldnn::qsa_indexer>().get_primitive();
    std::string reason;
    if (!ov::intel_gpu::op::QSAIndexer::validate_config(desc->get_config(), reason)) {
        return false;
    }

    if (desc->num_selection_heads != 1 || desc->indexer_head_dim % 16 != 0 || desc->indexer_head_dim % qsa::subgroup_size(info) != 0) {
        return false;
    }
    const auto& index_query_ps = node.get_input_layout(QSAIn::INDEX_QUERY).get_partial_shape();
    const auto& raw_index_keys_ps = node.get_input_layout(QSAIn::RAW_INDEX_KEYS).get_partial_shape();
    if (index_query_ps.rank().is_dynamic() || index_query_ps.size() != 4 || index_query_ps[2].is_dynamic()) {
        return false;
    }
    if (raw_index_keys_ps.rank().is_dynamic() || raw_index_keys_ps.size() != 4 || raw_index_keys_ps[2].is_dynamic() || raw_index_keys_ps[2].get_length() != 1) {
        return false;
    }

    return node.get_input_layout(QSAIn::INDEX_QUERY).data_type == ov::element::f16 &&
           node.get_input_layout(QSAIn::RAW_INDEX_KEYS).data_type == ov::element::f16;
}

std::unique_ptr<cldnn::primitive_impl> QSAIndexerImplementationManager::create_impl(const cldnn::program_node& node,
                                                                                    const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::qsa_indexer>());
    return std::make_unique<QSAIndexerImpl>(params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::QSAIndexerImpl)

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "../primitive_cm_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/qsa_indexer.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::cm {

using namespace cldnn;
using QSAIn = ov::intel_gpu::op::QSAIndexer::InputIdx;

namespace {

std::vector<int32_t> read_i32_input(const primitive_inst& instance, size_t input_idx) {
    auto mem = instance.input_memory_ptr(input_idx);
    mem_lock<int32_t, mem_lock_type::read> lock(mem, instance.get_network().get_stream());
    return {lock.begin(), lock.end()};
}

QSAIndexerRuntimeParams* rt_of(ImplRuntimeParams* rt) {
    auto* p = static_cast<QSAIndexerRuntimeParams*>(rt);
    OPENVINO_ASSERT(p != nullptr, "[GPU][CM] qsa_indexer runtime params are not initialized");
    return p;
}

/// H_iq / H_ik, derived from the 4D [B, L, H, D_idx] q_idx/k_idx layouts.
size_t index_heads_4d(const layout& l) {
    const auto& shape = l.get_shape();
    OPENVINO_ASSERT(shape.size() >= 3, "[GPU][CM] qsa_indexer: expected a rank-4 [B, L, H, D_idx] input");
    return shape[2];
}

JitConstants make_jit(const std::shared_ptr<const qsa_indexer>& desc, const RuntimeParams& params) {
    const auto h_iq = index_heads_4d(params.get_input_layout(QSAIn::Q_IDX));

    JitConstants jit;
    jit.make("QSA_IDX_QUERY_HEADS", h_iq);
    jit.make("QSA_IDX_HEAD_DIM", desc->indexer_head_dim);
    jit.make("QSA_IDX_ROTARY_DIM", desc->indexer_rotary_dim);
    jit.make("QSA_COMPRESS_RATIO", desc->compress_ratio);
    jit.make("QSA_BLOCK_TOPK", desc->block_topk);
    jit.make("QSA_ADDRESS_MODE", 1);
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
constexpr size_t dpas_k_step = 16;      // fp16 K elements per dpas
constexpr size_t max_slm_bytes = 64 * 1024;
constexpr size_t score_wg_size = 16;  // threads per query tile in the dpas Q2 score kernel

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
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::Q_IDX});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::K_IDX});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::Q_NORM_WEIGHT});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::K_NORM_WEIGHT});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::ROTARY_COS_SIN});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::POSITION_IDS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_RAW_K_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::INDEXER_POSITION_CACHE});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::INDEXER_Q});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::PREPARE_MAP});
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
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->raw_k_stride);
            push_u32(kd, rtp->summary_stride);
        }};
    }
};

// ===========================================================================
// Q2 - score (dense addressing)
// ===========================================================================
class DenseScoreGenerator : public KernelGenerator {
public:
    DenseScoreGenerator() : KernelGenerator("qsa_indexer_score", "_cm") {}

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
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
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
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::SUMMARY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::SCORE_TILE_MAP});
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
    DenseTopkGenerator() : KernelGenerator("qsa_indexer_topk", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_qsa_indexer_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<qsa_indexer>(), params));
        jit.make("QSA_TOPK_WG", topk_wg_size);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, QSAIndexerInternalBufIdx::BLOCK_SCORES});
        args.push_back({ArgumentDescriptor::Types::INPUT, QSAIn::PAST_LENS});
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
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->max_blocks);
        }};
    }
};

class QSAIndexerCmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::QSAIndexerCmImpl)

    Stage::Ptr prepare = make_stage<DensePrepareGenerator>();
    Stage::Ptr score = make_stage<DenseScoreGenerator>();
    Stage::Ptr score_dpas = make_stage<DenseScoreDpasGenerator>();
    Stage::Ptr topk = make_stage<DenseTopkGenerator>();

    QSAIndexerCmImpl() : PrimitiveImplCM(QSAIndexerCMImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<QSAIndexerRuntimeParams>();
    }

    QSAIndexerRuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<QSAIndexerRuntimeParams>();
        }
        return *static_cast<QSAIndexerRuntimeParams*>(m_rt_params.get());
    }

    explicit QSAIndexerCmImpl(const RuntimeParams& params) : QSAIndexerCmImpl() {
        add_stage(prepare, params);
        add_stage(score, params);
        if (use_dpas_score(params)) {
            add_stage(score_dpas, params);
        }
        add_stage(topk, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<QSAIndexerCmImpl>(this);
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
        const auto desc = params.typed_desc<qsa_indexer>();
        const auto h_iq = index_heads_4d(params.get_input_layout(QSAIn::Q_IDX));
        return can_use_dpas_score(*desc, h_iq, params.get_device_info());
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        const auto desc = params.typed_desc<qsa_indexer>();
        const auto& q_idx_shape = params.get_input_layout(QSAIn::Q_IDX).get_shape();
        const auto batch = std::max<size_t>(1, q_idx_shape[0]);
        const auto seq_len = std::max<size_t>(1, q_idx_shape[1]);
        const auto h_iq = index_heads_4d(params.get_input_layout(QSAIn::Q_IDX));
        const auto tokens = batch * seq_len;

        const auto& raw_shape = params.get_input_layout(QSAIn::INDEXER_RAW_K_CACHE).get_shape();
        const auto s_cap = std::max<size_t>(1, raw_shape[2]);
        const auto max_blocks = std::max<size_t>(1, s_cap / desc->compress_ratio);
        const auto prepare_pairs = tokens + batch;
        // One 16-query tile per batch row, upper-bounded (see qsa.cpp's identical bound).
        const auto score_tiles = tokens / dpas_query_tile + batch + 1;

        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(prepare_pairs * 2, ov::element::i32, true, false);
        bufs.emplace_back(tokens * h_iq * desc->indexer_head_dim, ov::element::f16);
        bufs.emplace_back(tokens * max_blocks, ov::element::f32);
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
        const auto desc = params.typed_desc<qsa_indexer>();
        const auto past_lens = read_i32_input(instance, QSAIn::PAST_LENS);

        const auto& q_idx_layout = params.get_input_layout(QSAIn::Q_IDX);
        const auto& q_idx_shape = q_idx_layout.get_shape();
        const size_t batch = q_idx_shape[0];
        const size_t seq_len = q_idx_shape[1];
        OPENVINO_ASSERT(past_lens.size() == batch, "[GPU][CM] qsa_indexer: past_lens must have B entries");

        rtp->batch = batch;
        rtp->seq_len = seq_len;
        rtp->num_tokens = batch * seq_len;
        rtp->index_query_heads = index_heads_4d(q_idx_layout);
        rtp->index_key_heads = index_heads_4d(params.get_input_layout(QSAIn::K_IDX));

        const auto& raw_shape = params.get_input_layout(QSAIn::INDEXER_RAW_K_CACHE).get_shape();
        const auto& summary_shape = params.get_input_layout(QSAIn::SUMMARY_CACHE).get_shape();
        rtp->raw_k_stride = raw_shape[2];
        rtp->summary_stride = summary_shape[2];

        const auto r = desc->compress_ratio;
        size_t max_blocks = 1;
        rtp->prepare_map.clear();
        for (size_t b = 0; b < batch; b++) {
            const auto past_len = static_cast<size_t>(std::max(0, past_lens[b]));
            const size_t total_len = past_len + seq_len;
            max_blocks = std::max(max_blocks, total_len / r);

            const size_t first_block = past_len / r;
            const size_t last_block = (total_len - 1) / r;
            for (size_t block = first_block; block <= last_block; block++) {
                rtp->prepare_map.push_back(static_cast<int32_t>(b));
                rtp->prepare_map.push_back(static_cast<int32_t>(block));
            }
        }
        rtp->prepare_wg_count = rtp->prepare_map.size() / 2;
        rtp->max_blocks = max_blocks;

        // Score tile map (dpas score only; harmless to build unconditionally).
        rtp->score_tile_map.clear();
        for (size_t b = 0; b < batch; b++) {
            for (size_t t = 0; t < seq_len; t += dpas_query_tile) {
                rtp->score_tile_map.push_back(static_cast<int32_t>(b));
                rtp->score_tile_map.push_back(static_cast<int32_t>(t));  // local position within the batch row
            }
        }
        rtp->score_tile_count = rtp->score_tile_map.size() / 2;
    }

private:
    void upload_prepare_map(primitive_inst& instance) {
        auto* rtp = &rt();
        if (!rtp->prepare_map.empty()) {
            auto mem = instance.get_intermediates_memories()[QSAIndexerInternalBufIdx::PREPARE_MAP];
            OPENVINO_ASSERT(mem->count() >= rtp->prepare_map.size(), "[GPU][CM] qsa_indexer prepare map buffer is too small");
            mem_lock<int32_t, mem_lock_type::write> lock(mem, instance.get_network().get_stream());
            std::copy(rtp->prepare_map.begin(), rtp->prepare_map.end(), lock.begin());
        }
        if (!rtp->score_tile_map.empty() && use_dpas_score(*instance.get_impl_params())) {
            auto mem = instance.get_intermediates_memories()[QSAIndexerInternalBufIdx::SCORE_TILE_MAP];
            OPENVINO_ASSERT(mem->count() >= rtp->score_tile_map.size(), "[GPU][CM] qsa_indexer score tile map buffer is too small");
            mem_lock<int32_t, mem_lock_type::write> lock(mem, instance.get_network().get_stream());
            std::copy(rtp->score_tile_map.begin(), rtp->score_tile_map.end(), lock.begin());
        }
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

    if (desc->h_sel != 1) {
        return false;
    }
    const auto h_ik = index_heads_4d(node.get_input_layout(QSAIn::K_IDX));
    if (h_ik != 1) {
        return false;
    }

    const auto& q_idx_shape = node.get_input_layout(QSAIn::Q_IDX).get_shape();
    if (q_idx_shape.size() != 4) {
        return false;
    }

    if (node.get_input_layout(QSAIn::Q_IDX).data_type != ov::element::f16 || node.get_input_layout(QSAIn::K_IDX).data_type != ov::element::f16) {
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


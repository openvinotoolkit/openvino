// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_sdpa.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "../primitive_cm_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/sparse_sdpa.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::cm {

using namespace cldnn;
using SparseSDPAIn = ov::intel_gpu::op::SparseSDPA::InputIdx;

namespace {

SparseSDPARuntimeParams* rt_of(ImplRuntimeParams* rt) {
    auto* p = static_cast<SparseSDPARuntimeParams*>(rt);
    OPENVINO_ASSERT(p != nullptr, "[GPU][CM] sparse_sdpa runtime params are not initialized");
    return p;
}

JitConstants make_jit(const std::shared_ptr<const sparse_sdpa>& desc, const RuntimeParams& params) {
    const auto& sel_shape = params.get_input_layout(SparseSDPAIn::SEL_INDICES).get_shape();
    OPENVINO_ASSERT(sel_shape.size() >= 2, "[GPU][CM] sparse_sdpa expects a rank>=2 sel_indices input");

    JitConstants jit;
    jit.make("QSA_HEADS", desc->num_heads);
    jit.make("QSA_KV_HEADS", desc->num_kv_heads);
    jit.make("QSA_K_HEAD_SIZE", desc->k_head_size);
    jit.make("QSA_V_HEAD_SIZE", desc->v_head_size);
    jit.make("QSA_COMPRESS_RATIO", desc->sel_block_size);
    jit.make("QSA_BLOCK_TOPK", sel_shape.back());
    jit.add(make_jit_constant("QSA_SCALE", desc->scale));
    return jit;
}

void push_u32(KernelData& kd, size_t value) {
    scalar_desc s;
    s.t = scalar_desc::Types::UINT32;
    s.v.u32 = static_cast<uint32_t>(value);
    kd.params.scalars.push_back(s);
}

constexpr auto get_sparse_attention_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
}

constexpr size_t dpas_query_tile = 16;  // dpas N axis == GRF width on Xe2
constexpr size_t dpas_k_step = 16;      // fp16 K elements per dpas
constexpr size_t dpas_wide_cols = 32;   // halfs per 2D block row (64-byte hardware cap)

/// Heads that share one dpas tile. See qsa.cpp's dpas_heads_per_tile() for the rationale.
size_t dpas_heads_per_tile(const sparse_sdpa& desc) {
    const auto n_rep = desc.num_heads / desc.num_kv_heads;
    for (size_t ht : {size_t{16}, size_t{8}, size_t{4}, size_t{2}}) {
        if (n_rep % ht == 0) {
            return ht;
        }
    }
    return 1;
}

/// Threads that cooperate on one (query tile, head). See qsa.cpp's dpas_attention_workers().
size_t dpas_attention_workers(const sparse_sdpa& desc) {
    constexpr size_t reg_budget = 12 * 1024;
    for (size_t w : {size_t{1}, size_t{2}, size_t{4}, size_t{8}}) {
        if (desc.k_head_size % (w * dpas_k_step) != 0 || desc.v_head_size % (w * dpas_query_tile) != 0) {
            continue;
        }
        const auto dh_w = desc.k_head_size / w;
        const auto dv_w = desc.v_head_size / w;
        if (dh_w % dpas_wide_cols != 0 || dv_w % dpas_wide_cols != 0) {
            continue;
        }
        const auto rq = dh_w * dpas_query_tile * sizeof(ov::float16);
        const auto ro = dpas_query_tile * dv_w * sizeof(float);
        const auto kv = dpas_k_step * (dh_w + dv_w) * sizeof(ov::float16);
        if (rq + ro + kv <= reg_budget) {
            return w;
        }
    }
    return 0;
}

/// Whether the systolic sparse_attention_dense_dpas kernel can be used. Unlike the
/// paged capability gate, there is no page/compress-ratio page-boundary check at all:
/// dense K/V for one (batch, kv_head) is one contiguous region.
bool can_use_dpas_attention(const sparse_sdpa& desc, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;
    }
    if (desc.k_head_size % dpas_k_step != 0 || desc.v_head_size % dpas_query_tile != 0) {
        return false;
    }
    if (desc.k_head_size * sizeof(ov::float16) < 64 || desc.v_head_size * sizeof(ov::float16) < 64) {
        return false;
    }
    if (desc.sel_block_size % 2 != 0 || dpas_k_step % desc.sel_block_size != 0) {
        return false;  // four QSA blocks must pack exactly one 16-token dpas K step
    }
    return dpas_attention_workers(desc) != 0;
}

/// Whether the GQA-aware systolic sparse_attention_dense_gqa_dpas kernel can be used.
/// Preferred over the plain dpas kernel whenever it holds - this is what matters for GQA
/// ratios like Qwen4's Hq/Hkv = 12 (see qsa.cpp's identical can_use_gqa_dpas_attention()).
bool can_use_gqa_dpas_attention(const sparse_sdpa& desc, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;
    }
    if (desc.num_kv_heads == 0 || desc.num_heads % desc.num_kv_heads != 0) {
        return false;
    }
    const auto n_rep = desc.num_heads / desc.num_kv_heads;
    if (n_rep < 1 || n_rep > 16) {
        return false;
    }
    constexpr size_t gqa_workers = 4;  // GQA_W in sparse_attention_dense_gqa_dpas.cm
    if (desc.k_head_size % (gqa_workers * 32) != 0 || desc.v_head_size % (gqa_workers * 32) != 0) {
        return false;
    }
    if (std::max(desc.k_head_size, desc.v_head_size) > 256) {
        return false;
    }
    if (desc.sel_block_size != 2 && desc.sel_block_size != 4 && desc.sel_block_size != 8 && desc.sel_block_size != 16) {
        return false;
    }
    return desc.causal;
}

// ===========================================================================
// Q3 - selected-block sparse attention (dense, read-only main K/V)
// ===========================================================================
class AttentionDenseGenerator : public KernelGenerator {
public:
    AttentionDenseGenerator() : KernelGenerator("sparse_attention_dense", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<sparse_sdpa>(), params));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::KEY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::VALUE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SEL_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SEL_COUNT});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
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
            const auto desc = params.typed_desc<sparse_sdpa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->num_heads, std::max<size_t>(1, rtp->batch * rtp->seq_len), 1};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->history_len);
            push_u32(kd, rtp->batch * rtp->seq_len);
        }};
    }
};

// ===========================================================================
// Q3 prefill, systolic variant - selected-block sparse attention (dense) on cm_dpas
// ===========================================================================
class AttentionDenseDpasGenerator : public KernelGenerator {
public:
    AttentionDenseDpasGenerator() : KernelGenerator("sparse_attention_dense_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<sparse_sdpa>();
        jit.add(make_jit(desc, params));
        jit.make("QSA_ATT_WORKERS", std::max<size_t>(1, dpas_attention_workers(*desc)));
        jit.make("QSA_HEADS_PER_TILE", dpas_heads_per_tile(*desc));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::KEY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::VALUE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SEL_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SEL_COUNT});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparseSDPAInternalBufIdx::TILE_MAP});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
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
            const auto desc = params.typed_desc<sparse_sdpa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            const size_t workers = std::max<size_t>(1, dpas_attention_workers(*desc));
            const size_t heads_per_tile = dpas_heads_per_tile(*desc);
            wgs.global = {(desc->num_heads / heads_per_tile) * workers, std::max<size_t>(1, rtp->tile_count) * heads_per_tile, 1};
            wgs.local = {workers, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->tile_count);
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->history_len);
        }};
    }
};

// ===========================================================================
// Q3 prefill, GQA-aware systolic variant - one token x one KV head per WG on cm_dpas
// ===========================================================================
class AttentionDenseGqaDpasGenerator : public KernelGenerator {
public:
    AttentionDenseGqaDpasGenerator() : KernelGenerator("sparse_attention_dense_gqa_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<sparse_sdpa>();
        jit.add(make_jit(desc, params));
        jit.make("QSA_IS_CAUSAL", desc->causal ? 1 : 0);
        jit.make("QSA_GQA_HEAD_MAJOR", 1);
        jit.make("QSA_GQA_VECTOR_META", 1);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::KEY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::VALUE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SEL_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SEL_COUNT});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparseSDPAInternalBufIdx::TOKEN_MAP});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
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
            const auto desc = params.typed_desc<sparse_sdpa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            // QSA_GQA_HEAD_MAJOR=1: dim0 enumerates (token, W4 worker), dim1 the KV head.
            const size_t tokens = std::max<size_t>(1, rtp->batch * rtp->seq_len);
            wgs.global = {tokens * 4, desc->num_kv_heads, 1};
            wgs.local = {4, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, tokens);
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->history_len);
        }};
    }
};

class SparseSDPACmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::SparseSDPACmImpl)

    Stage::Ptr attention = make_stage<AttentionDenseGenerator>();
    Stage::Ptr attention_dpas = make_stage<AttentionDenseDpasGenerator>();
    Stage::Ptr attention_gqa_dpas = make_stage<AttentionDenseGqaDpasGenerator>();

    SparseSDPACmImpl() : PrimitiveImplCM(SparseSDPACMImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<SparseSDPARuntimeParams>();
    }

    SparseSDPARuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<SparseSDPARuntimeParams>();
        }
        return *static_cast<SparseSDPARuntimeParams*>(m_rt_params.get());
    }

    explicit SparseSDPACmImpl(const RuntimeParams& params) : SparseSDPACmImpl() {
        add_stage(attention, params);
        if (use_gqa_dpas_attention(params)) {
            add_stage(attention_gqa_dpas, params);
        } else if (use_dpas_attention(params)) {
            add_stage(attention_dpas, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SparseSDPACmImpl>(this);
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
    static bool use_dpas_attention(const kernel_impl_params& params) {
        return can_use_dpas_attention(*params.typed_desc<sparse_sdpa>(), params.get_device_info());
    }
    static bool use_gqa_dpas_attention(const kernel_impl_params& params) {
        return can_use_gqa_dpas_attention(*params.typed_desc<sparse_sdpa>(), params.get_device_info());
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        const auto& query_shape = params.get_input_layout(SparseSDPAIn::QUERY).get_shape();
        const auto batch = std::max<size_t>(1, query_shape[0]);
        const auto seq_len = std::max<size_t>(1, query_shape[2]);
        // One 16-query tile per batch row, upper-bounded (see qsa.cpp's identical bound).
        const auto tiles = (batch * seq_len) / dpas_query_tile + batch + 1;

        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(tiles * 2, ov::element::i32, true, false);            // TILE_MAP (host-filled)
        bufs.emplace_back(batch * seq_len * 2, ov::element::i32, true, false);  // TOKEN_MAP (host-filled)
        return bufs;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        update_rt_params(instance);
        const auto& params = *instance.get_impl_params();
        if (use_gqa_dpas_attention(params)) {
            upload_token_map(instance);
            return execute_stage(events, instance, attention_gqa_dpas);
        }
        if (use_dpas_attention(params)) {
            upload_tile_map(instance);
            return execute_stage(events, instance, attention_dpas);
        }
        return execute_stage(events, instance, attention);
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        auto* rtp = &rt();

        const auto& params = *instance.get_impl_params();
        const auto& query_shape = params.get_input_layout(SparseSDPAIn::QUERY).get_shape();
        const auto& key_shape = params.get_input_layout(SparseSDPAIn::KEY).get_shape();
        OPENVINO_ASSERT(query_shape.size() == 4 && key_shape.size() == 4,
                        "[GPU][CM] sparse_sdpa expects rank-4 [B,H,*,D] query/key inputs");

        rtp->batch = query_shape[0];
        rtp->seq_len = query_shape[2];
        rtp->history_len = key_shape[2];

        rtp->tile_map.clear();
        rtp->token_map.clear();
        if (use_gqa_dpas_attention(params)) {
            for (size_t b = 0; b < rtp->batch; b++) {
                for (size_t t = 0; t < rtp->seq_len; t++) {
                    rtp->token_map.push_back(static_cast<int32_t>(b));
                    rtp->token_map.push_back(static_cast<int32_t>(t));
                }
            }
        } else if (use_dpas_attention(params)) {
            for (size_t b = 0; b < rtp->batch; b++) {
                for (size_t t = 0; t < rtp->seq_len; t += dpas_query_tile) {
                    rtp->tile_map.push_back(static_cast<int32_t>(b));
                    rtp->tile_map.push_back(static_cast<int32_t>(t));
                }
            }
        }
        rtp->tile_count = rtp->tile_map.size() / 2;
    }

private:
    void upload_tile_map(primitive_inst& instance) {
        auto* rtp = &rt();
        if (rtp->tile_map.empty()) {
            return;
        }
        auto mem = instance.get_intermediates_memories()[SparseSDPAInternalBufIdx::TILE_MAP];
        OPENVINO_ASSERT(mem->count() >= rtp->tile_map.size(), "[GPU][CM] sparse_sdpa tile map buffer is too small");
        mem_lock<int32_t, mem_lock_type::write> lock(mem, instance.get_network().get_stream());
        std::copy(rtp->tile_map.begin(), rtp->tile_map.end(), lock.begin());
    }

    void upload_token_map(primitive_inst& instance) {
        auto* rtp = &rt();
        if (rtp->token_map.empty()) {
            return;
        }
        auto mem = instance.get_intermediates_memories()[SparseSDPAInternalBufIdx::TOKEN_MAP];
        OPENVINO_ASSERT(mem->count() >= rtp->token_map.size(), "[GPU][CM] sparse_sdpa token map buffer is too small");
        mem_lock<int32_t, mem_lock_type::write> lock(mem, instance.get_network().get_stream());
        std::copy(rtp->token_map.begin(), rtp->token_map.end(), lock.begin());
    }
};

}  // namespace

bool SparseSDPACMImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::sparse_sdpa>());

    if (node.has_fused_primitives()) {
        return false;
    }

    auto& engine = node.get_program().get_engine();
    const auto& config = node.get_program().get_config();
    const auto& info = engine.get_device_info();
    if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
        return false;
    }

    const auto desc = node.as<cldnn::sparse_sdpa>().get_primitive();
    std::string reason;
    if (!ov::intel_gpu::op::SparseSDPA::validate_config(desc->get_config(), reason)) {
        return false;
    }

    if (desc->h_sel != 1) {
        return false;
    }

    static constexpr std::array f16_inputs = {SparseSDPAIn::QUERY, SparseSDPAIn::KEY, SparseSDPAIn::VALUE};
    for (auto idx : f16_inputs) {
        if (node.get_input_layout(idx).data_type != ov::element::f16) {
            return false;
        }
    }
    if (node.get_input_layout(SparseSDPAIn::SEL_INDICES).data_type != ov::element::i32 ||
        node.get_input_layout(SparseSDPAIn::SEL_COUNT).data_type != ov::element::i32) {
        return false;
    }

    return true;
}

std::unique_ptr<cldnn::primitive_impl> SparseSDPACMImplementationManager::create_impl(const cldnn::program_node& node,
                                                                                       const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::sparse_sdpa>());
    return std::make_unique<SparseSDPACmImpl>(params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sparse_sdpa)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::SparseSDPACmImpl)

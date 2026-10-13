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

class SparseSDPACmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::SparseSDPACmImpl)

    Stage::Ptr attention = make_stage<AttentionDenseGenerator>();

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

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& /*params*/) const override {
        return {};
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        update_rt_params(instance);
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

// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vl_sdpa_opt.hpp"

#include <algorithm>
#include <cmath>

#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/primitives/gated_delta_net.hpp"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"
#include "gated_delta_net_opt.hpp"

namespace ov::intel_gpu::cm {
namespace {

constexpr auto get_linear_attention_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
}

class GatedDeltaNetGenerator : public KernelGenerator {
public:
    GatedDeltaNetGenerator() : KernelGenerator("recurrent_gated_delta_rule") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_linear_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        auto desc = params.typed_desc<gated_delta_net>();
        const auto query_shape = params.get_input_layout(0).get_partial_shape();
        const auto key_shape = params.get_input_layout(1).get_partial_shape();
        const auto value_shape = params.get_input_layout(2).get_partial_shape();

        const size_t k_head_size = key_shape[query_shape.size() - 1].get_length();
        const size_t v_head_size = value_shape[query_shape.size() - 1].get_length();
        const size_t num_k_heads = query_shape[query_shape.size() - 3].get_length();
        const size_t num_v_heads = value_shape[key_shape.size() - 3].get_length();
        const float scale_factor = 1.0 / std::sqrt(static_cast<double>(k_head_size));

        GPU_DEBUG_TRACE_DETAIL << "GatedDeltaNet query_shape " << query_shape << ", key_shape " << key_shape << ", k_head_size="
                               << k_head_size << ", v_head_size=" << v_head_size << ", num_k_heads=" << num_k_heads << ", num_v_heads=" << num_v_heads << '\n';

        jit.add({
            make_jit_constant("KERNEL_NAME", get_entry_point(params)),
            make_jit_constant("K_HEAD_NUMS", num_k_heads),
            make_jit_constant("V_HEAD_NUMS", num_v_heads),
            make_jit_constant("K_HEAD_DIMS", k_head_size),
            make_jit_constant("V_HEAD_DIMS", num_v_heads),
            make_jit_constant("SCALE_FACTOR", scale_factor),
        });

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        for (uint32_t i = 0; i < params.input_layouts.size() - 1; i++) {  // inputs: q, k, v
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto desc = params.typed_desc<gated_delta_net>();
            const auto query_shape = params.get_input_layout(0).get_shape();
            // B, T, H, K
            const size_t batch = query_shape[0];
            const size_t head_nums = query_shape[2];

            auto& wgs = kd.params.workGroups;
            wgs.global = {batch, head_nums, 1};
            wgs.local = {1, 1, 8};
        }};
    }
};

class GatedDeltaNetOptImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::GatedDeltaNetOptImpl)

    Stage::Ptr gated_delta_net = make_stage<GatedDeltaNetGenerator>();
    GatedDeltaNetOptImpl() : PrimitiveImplOCL(GatedDeltaNetOptImplementationManager::get_type_info_static()) {
    }
    GatedDeltaNetOptImpl(const program_node& node, const RuntimeParams& params) : GatedDeltaNetOptImpl() {
        add_stage(gated_delta_net, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatedDeltaNetOptImpl>(this);
    }

    void update_rt_params(const cldnn::primitive_inst& instance) override {
        update_stages_flags(instance);
        return;
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        return PrimitiveImplCM::execute(events, instance);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GatedDeltaNetOptImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<gated_delta_net>());
    return std::make_unique<GatedDeltaNetOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gated_delta_net)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::GatedDeltaNetOptImpl)

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

constexpr auto get_gated_delta_net_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
}

class GatedDeltaNetGenerator : public KernelGenerator {
public:
    GatedDeltaNetGenerator() : KernelGenerator("cm_gated_delta_net") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_gated_delta_net_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        auto desc = params.typed_desc<gated_delta_net>();

        const auto query_shape = params.get_input_layout(0).get_partial_shape();
        const auto key_shape = params.get_input_layout(1).get_partial_shape();
        const auto value_shape = params.get_input_layout(2).get_partial_shape();

        const size_t k_head_size = key_shape[query_shape.size() - 1].get_length();
        const size_t v_head_size = value_shape[query_shape.size() - 1].get_length();
        const size_t num_k_heads = query_shape[query_shape.size() - 2].get_length();
        const size_t num_v_heads = value_shape[key_shape.size() - 2].get_length();
        const float scale_factor = 1.0 / std::sqrt(static_cast<double>(k_head_size));
        const auto io_type = params.get_input_layout(0).data_type == data_types::f16 ? 0 : 1;
        const auto num_outputs = desc->output_size();

        GPU_DEBUG_TRACE_DETAIL << "GatedDeltaNet io_type" << io_type << " query_shape " << query_shape << ", key_shape " << key_shape
                               << ", k_head_size=" << k_head_size << ", v_head_size=" << v_head_size << ", num_k_heads=" << num_k_heads
                               << ", num_v_heads=" << num_v_heads << ", num_outputs=" << num_outputs << '\n';
        jit.add({
            make_jit_constant("KERNEL_NAME", get_entry_point(params)),
            make_jit_constant("K_HEAD_NUMS", num_k_heads),
            make_jit_constant("V_HEAD_NUMS", num_v_heads),
            make_jit_constant("K_HEAD_DIMS", k_head_size),
            make_jit_constant("V_HEAD_DIMS", v_head_size),
            make_jit_constant("SCALE_FACTOR", scale_factor),
            make_jit_constant("IO_TYPE", io_type),
        });

        if (params.output_layouts.size() > 1) {
            jit.add(make_jit_constant("OUTPUT_STATE", 1));
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {  // inputs: q, k, v, initial_state, g, beta
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            const auto query_shape = params.get_input_layout(0).get_shape();
            const auto value_shape = params.get_input_layout(2).get_shape();
            // B, T, H, K
            const size_t batch = query_shape[0];
            const size_t seq = query_shape[1];
            const size_t head_nums = value_shape[2];

            const auto& info = params.get_device_info();
            const size_t thread_nums = (info.arch <= gpu_arch::xe_hpc) ? 16 : 8;
            auto& wgs = kd.params.workGroups;
            wgs.global = {batch, head_nums, thread_nums};
            wgs.local = {1, 1, thread_nums};
            auto key_layout = params.input_layouts[1];
            auto value_layout = params.input_layouts[2];
            auto get_simple_offset = [](const cldnn::layout& layout) {
                size_t offset = 0;
                const auto& data_padding = layout.data_padding;
                const auto& lower_pads = data_padding._lower_size;
                for (auto& it : lower_pads) {
                    if (it > 0) {
                        offset = it;
                        break;
                    }
                }
                return offset;
            };
            size_t key_offset = get_simple_offset(key_layout);
            size_t value_offset = get_simple_offset(value_layout);
            std::vector<int32_t> scalars{static_cast<int32_t>(seq), static_cast<int32_t>(key_offset), static_cast<int32_t>(value_offset)};
            kd.params.scalars.clear();
            for (auto i : scalars) {
                scalar_desc desc;
                desc.t = scalar_desc::Types::INT32;
                desc.v.s32 = static_cast<int32_t>(i);
                kd.params.scalars.push_back(desc);
            }
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
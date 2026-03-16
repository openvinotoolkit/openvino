// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net_ref.hpp"

#include "intel_gpu/primitives/gated_delta_net.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

constexpr size_t v_block_size = 4;

size_t get_subgroup_size(gpu_arch arch) {
    switch (arch) {
    case gpu_arch::gen9:
    case gpu_arch::gen11:
    case gpu_arch::xe_lp:
    case gpu_arch::xe_hp:
    case gpu_arch::xe_hpg:
        return 8;
    case gpu_arch::xe_hpc:
    case gpu_arch::xe2:
    case gpu_arch::xe3:
        return 16;
    default:
        return 0;
    }
}

class GatedDeltaNetRefGenerator : public KernelGenerator {
public:
    GatedDeltaNetRefGenerator() : KernelGenerator("gated_delta_net_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto& q_shape = params.get_input_layout(0).get_partial_shape();
        const size_t q_head_nums = q_shape[2].get_length();
        const size_t k_head_dims = q_shape[3].get_length();
        const auto& v_shape = params.get_input_layout(2).get_partial_shape();
        const size_t v_head_nums = v_shape[2].get_length();
        const auto io_type = params.get_input_layout(0).data_type == data_types::f16 ? 0 : 1;
        const float scale_factor = 1.0f / std::sqrt(static_cast<double>(k_head_dims));
        const auto output_state = params.output_layouts.size() > 1 ? 1 : 0;

        jit.make("K_HEAD_NUM", q_head_nums);
        jit.make("V_HEAD_NUM", v_head_nums);
        jit.make("K_HEAD_DIM", k_head_dims);
        jit.make("SUBGROUP_SIZE", get_subgroup_size(params.get_device_info().arch));
        jit.make("IO_TYPE", io_type);
        jit.make("SCALE_FACTOR", scale_factor);
        jit.make("OUTPUT_STATE", output_state);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
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
            auto& wgs = kd.params.workGroups;

            const auto& q_shape = params.get_input_layout(0).get_partial_shape();
            const auto& v_shape = params.get_input_layout(2).get_partial_shape();
            const size_t batch = q_shape[0].get_length();
            const size_t seq_len = q_shape[1].get_length();
            const size_t head_nums = v_shape[2].get_length();
            const size_t k_head_dims = q_shape[3].get_length();
            const size_t v_blocks = (k_head_dims + v_block_size - 1) / v_block_size;
            const size_t subgroup_size = get_subgroup_size(params.get_device_info().arch);

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

            size_t key_offset = get_simple_offset(params.input_layouts[1]);
            size_t value_offset = get_simple_offset(params.input_layouts[2]);

            wgs.global = {batch, head_nums, v_blocks * subgroup_size};
            wgs.local = {1, 1, subgroup_size};

            kd.params.scalars.clear();
            std::vector<int32_t> scalars{static_cast<int32_t>(seq_len), static_cast<int32_t>(key_offset), static_cast<int32_t>(value_offset)};
            for (auto i : scalars) {
                scalar_desc desc;
                desc.t = scalar_desc::Types::INT32;
                desc.v.s32 = static_cast<int32_t>(i);
                kd.params.scalars.push_back(desc);
            }
        }};
    }
};

class GatedDeltaNetRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GatedDeltaNetRefImpl)

    Stage::Ptr gated_delta_net = make_stage<GatedDeltaNetRefGenerator>();

    GatedDeltaNetRefImpl() : PrimitiveImplOCL(GatedDeltaNetRef::get_type_info_static()) {}
    GatedDeltaNetRefImpl(const program_node& node, const RuntimeParams& params) : GatedDeltaNetRefImpl() {
        add_stage(gated_delta_net, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatedDeltaNetRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GatedDeltaNetRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<gated_delta_net>());
    return std::make_unique<GatedDeltaNetRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GatedDeltaNetRefImpl)
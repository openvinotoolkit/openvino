// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "fused_conv_ref.hpp"

#include "intel_gpu/primitives/fused_conv.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class FusedConvRefGenerator : public KernelGenerator {
public:
    FusedConvRefGenerator() : KernelGenerator("fused_conv_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto& input_shape = params.get_input_layout(0).get_partial_shape();
        const auto& weight_shape = params.get_input_layout(1).get_partial_shape();

        const size_t conv_dim = input_shape[1].get_length();
        const size_t kernel_size = weight_shape[1].get_length();
        const auto io_type = params.get_input_layout(0).data_type == data_types::f16 ? 0 : 1;

        jit.make("CONV_DIM", conv_dim);
        jit.make("KERNEL_SIZE", kernel_size);
        jit.make("IO_TYPE", io_type);

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
        // seq_len scalar
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto& input_shape = params.get_input_layout(0).get_partial_shape();
            const size_t batch = input_shape[0].get_length();
            const size_t conv_dim = input_shape[1].get_length();
            const size_t seq_len = input_shape[2].get_length();

            // Each work-item handles one (batch, channel) pair
            wgs.global = {batch, conv_dim, 1};
            wgs.local = {1, 256, 1};

            // Clamp local size to not exceed global
            if (wgs.local[1] > conv_dim) {
                wgs.local[1] = conv_dim;
            }

            kd.params.scalars.clear();
            scalar_desc desc;
            desc.t = scalar_desc::Types::INT32;
            desc.v.s32 = static_cast<int32_t>(seq_len);
            kd.params.scalars.push_back(desc);
        }};
    }
};

class FusedConvRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::FusedConvRefImpl)

    Stage::Ptr fused_conv_stage = make_stage<FusedConvRefGenerator>();

    FusedConvRefImpl() : PrimitiveImplOCL(FusedConvRef::get_type_info_static()) {}
    FusedConvRefImpl(const program_node& node, const RuntimeParams& params) : FusedConvRefImpl() {
        add_stage(fused_conv_stage, params);
    }

    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        auto args = PrimitiveImplOCL::get_arguments(instance);
        const auto& desc = instance.get_typed_desc<fused_conv>();
        auto& variable = instance.get_network().get_variable(desc->variable_info.variable_id);

        OPENVINO_ASSERT(args.inputs.size() >= 4, "[GPU] fused_conv expects 4 inputs");
        OPENVINO_ASSERT(args.outputs.size() >= 2, "[GPU] fused_conv expects 2 outputs");

        // Reuse variable state once initialized; otherwise use initial_state input.
        if (variable.is_set()) {
            args.inputs[3] = variable.get_memory();
        }
        // Write updated state directly to variable memory and avoid explicit Assign.
        args.outputs[1] = variable.get_memory();

        return args;
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        const auto& desc = instance.get_typed_desc<fused_conv>();
        auto& variable = instance.get_network().get_variable(desc->variable_info.variable_id);

        // Keep variable layout in sync with runtime state shape.
        variable.set_layout(instance.input_memory_ptr(3)->get_layout());

        // State source can change from initial_state -> variable after first iteration.
        for (const auto& stage_id : _order) {
            _stages[stage_id]->kd.need_args_update = true;
        }

        auto ev = PrimitiveImplOCL::execute(events, instance);
        variable.set();
        return ev;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<FusedConvRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> FusedConvRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<fused_conv>());
    return std::make_unique<FusedConvRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::FusedConvRefImpl)

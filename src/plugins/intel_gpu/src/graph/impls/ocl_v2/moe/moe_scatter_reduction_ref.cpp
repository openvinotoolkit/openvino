// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "moe_scatter_reduction_ref.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "../common_utils/jitter.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "intel_gpu/primitives/moe_scatter_reduction.hpp"

namespace ov::intel_gpu::ocl {
namespace {

using namespace ov::intel_gpu::ocl;
class MoeScatterReductionRefGenerator : public KernelGenerator {
public:
    MoeScatterReductionRefGenerator() : KernelGenerator("moe_scatter_reduction_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto in_l = params.input_layouts[0];
        auto hidden_size = extract_channel(ChannelName::Y, in_l);

        const auto& desc = params.typed_desc<moe_scatter_reduction>();

        jit.make("ACTIVE_EXPERTS", desc->num_active_experts_per_token);
        jit.make("HIDDEN_SIZE", hidden_size);
        return jit;
    }

    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        uint32_t num_of_inputs = 7;

        for (uint32_t i = 0; i < num_of_inputs; i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;

            if (!params.is_dynamic()) {
                auto num_tokens = extract_channel(ChannelName::BATCH, params.input_layouts[1]);
                wgs.global = {num_tokens, 1, 1};
                wgs.local = {1, 1, 1};
            }
        }};
    }
};

class MoeScatterReductionRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoeScatterReductionRefImpl)

    Stage::Ptr moe_scatter_reduction = make_stage<MoeScatterReductionRefGenerator>();

    MoeScatterReductionRefImpl() : PrimitiveImplOCL(MoeScatterReductionRef::get_type_info_static()) {}
    MoeScatterReductionRefImpl(const program_node& node, const RuntimeParams& params) : MoeScatterReductionRefImpl() {
        add_stage(moe_scatter_reduction, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoeScatterReductionRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MoeScatterReductionRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_scatter_reduction>());
    return std::make_unique<MoeScatterReductionRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_scatter_reduction)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoeScatterReductionRefImpl)

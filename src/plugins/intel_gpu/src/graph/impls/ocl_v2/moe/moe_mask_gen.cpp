// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "moe_mask_gen.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "../common_utils/jitter.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "intel_gpu/primitives/moe_mask_gen.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class MoeMaskGenRefGenerator : public KernelGenerator {
public:
    MoeMaskGenRefGenerator() : KernelGenerator("moe_mask_gen") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        auto prim = params.typed_desc<moe_mask_gen>();
        jit.make("NUM_EXPERTS_PER_TOKEN", prim->num_experts_per_token);

        return jit;
    }

    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});

        const uint32_t num_of_outputs = 5;
        for (uint32_t i = 0; i < num_of_outputs; i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            if (!params.is_dynamic()) {
                auto num_total_experts = static_cast<size_t>(params.typed_desc<moe_mask_gen>()->num_total_experts);
                wgs.global = {num_total_experts, 1, 1};
                wgs.local = {num_total_experts, 1, 1};
            }
        }};
    }
};

class MoeMaskGenRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoeMaskGenRefImpl)

    Stage::Ptr moe_mask_gen = make_stage<MoeMaskGenRefGenerator>();

    MoeMaskGenRefImpl() : PrimitiveImplOCL(MoeMaskGenRef::get_type_info_static()) {}
    MoeMaskGenRefImpl(const program_node& node, const RuntimeParams& params) : MoeMaskGenRefImpl() {
        add_stage(moe_mask_gen, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoeMaskGenRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MoeMaskGenRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_mask_gen>());
    return std::make_unique<MoeMaskGenRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_mask_gen)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoeMaskGenRefImpl)

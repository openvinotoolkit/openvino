// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msda_opt.hpp"

#include "common_utils/dispatch_utils.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/msda.hpp"
#include "msda_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class MSDAOptGenerator : public KernelGenerator {
public:
    MSDAOptGenerator() : KernelGenerator("msda_opt") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.make("KERNEL_NAME", get_entry_point(params));
        return jit;
    }

    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        for (uint32_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto batch_size = params.get_input_layout(0).get_shape()[0];
            const auto num_queries = params.get_input_layout(3).get_shape()[1];
            const auto num_heads = params.get_input_layout(0).get_shape()[2];
            const auto embed_dims = params.get_input_layout(0).get_shape()[3];
            wgs.global = {1, 1, batch_size * num_queries * num_heads * embed_dims};
            wgs.local = {1, 1, 64};
        }};
    }
};

class MSDAOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MSDAOptImpl)

    Stage::Ptr msda = make_stage<MSDAOptGenerator>();

    MSDAOptImpl() : PrimitiveImplOCL(MSDAOptImplementationManager::get_type_info_static()) {}
    MSDAOptImpl(const program_node& node, const RuntimeParams& params) : MSDAOptImpl() {
        add_stage(msda, params);
    }
    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MSDAOptImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MSDAOptImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<msda>());
    return std::make_unique<MSDAOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::msda)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MSDAOptImpl)

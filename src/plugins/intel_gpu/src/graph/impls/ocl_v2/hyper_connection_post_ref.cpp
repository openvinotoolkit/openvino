// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hyper_connection_post_ref.hpp"

#include "intel_gpu/primitives/hyper_connection_post.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class HyperConnectionPostRefGenerator : public KernelGenerator {
public:
    HyperConnectionPostRefGenerator() : KernelGenerator("hyper_connection_post_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& shape = params.get_input_layout(1).get_partial_shape();
        const auto r = shape.size();
        jit.make("HC_N", shape[r - 2].get_length());
        jit.make("HC_E", shape[r - 1].get_length());
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            assert(!params.is_dynamic());
            const auto& shape = params.get_input_layout(1).get_shape();
            size_t rows = 1;
            for (size_t i = 0; i + 2 < shape.size(); i++) {
                rows *= shape[i];
            }
            // one work-item per (row, element); it writes all n streams
            auto& wgs = kd.params.workGroups;
            const size_t E = shape.back();
            const size_t lws = E % 16 == 0 ? 16 : 1;
            wgs.global = {E, rows, 1};
            wgs.local = {lws, 1, 1};
        }};
    }
};

class HyperConnectionPostRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::HyperConnectionPostRefImpl)

    Stage::Ptr stage = make_stage<HyperConnectionPostRefGenerator>();

    HyperConnectionPostRefImpl() : PrimitiveImplOCL(HyperConnectionPostRef::get_type_info_static()) {}
    HyperConnectionPostRefImpl(const program_node& node, const RuntimeParams& params) : HyperConnectionPostRefImpl() {
        add_stage(stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<HyperConnectionPostRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> HyperConnectionPostRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<hyper_connection_post>());
    return std::make_unique<HyperConnectionPostRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::hyper_connection_post)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::HyperConnectionPostRefImpl)

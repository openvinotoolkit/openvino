// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hyper_connection_mix_ref.hpp"

#include "intel_gpu/primitives/hyper_connection_mix.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class HyperConnectionMixRefGenerator : public KernelGenerator {
public:
    HyperConnectionMixRefGenerator() : KernelGenerator("hyper_connection_mix_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<hyper_connection_mix>();
        jit.make("HC_N", desc->n_streams);
        jit.make("HC_MIX", (2 + desc->n_streams) * desc->n_streams);
        jit.make("HC_ITERS", desc->sinkhorn_iterations);
        jit.make("HC_EPS", desc->eps);
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
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            assert(!params.is_dynamic());
            const auto& shape = params.get_input_layout(0).get_shape();
            size_t rows = 1;
            for (size_t i = 0; i + 1 < shape.size(); i++) {
                rows *= shape[i];
            }
            // n == 4: one 16-lane subgroup per token row (lane = one comb entry); otherwise one work-item per row
            const auto n = params.typed_desc<hyper_connection_mix>()->n_streams;
            auto& wgs = kd.params.workGroups;
            if (n == 4) {
                wgs.global = {rows * 16, 1, 1};
                wgs.local = {16, 1, 1};
            } else {
                const size_t lws = 16;
                wgs.global = {((rows + lws - 1) / lws) * lws, 1, 1};
                wgs.local = {lws, 1, 1};
            }
            kd.params.scalars.clear();
            scalar_desc s;
            s.t = scalar_desc::Types::INT32;
            s.v.s32 = static_cast<int32_t>(rows);
            kd.params.scalars.push_back(s);
        }};
    }
};

class HyperConnectionMixRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::HyperConnectionMixRefImpl)

    Stage::Ptr stage = make_stage<HyperConnectionMixRefGenerator>();

    HyperConnectionMixRefImpl() : PrimitiveImplOCL(HyperConnectionMixRef::get_type_info_static()) {}
    HyperConnectionMixRefImpl(const program_node& node, const RuntimeParams& params) : HyperConnectionMixRefImpl() {
        add_stage(stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<HyperConnectionMixRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> HyperConnectionMixRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<hyper_connection_mix>());
    return std::make_unique<HyperConnectionMixRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::hyper_connection_mix)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::HyperConnectionMixRefImpl)

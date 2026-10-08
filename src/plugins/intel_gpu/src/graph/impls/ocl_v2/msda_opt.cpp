// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msda_opt.hpp"

#include "common_utils/dispatch_utils.hpp"
#include "intel_gpu/primitives/msda.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class MSDAOptGenerator : public KernelGenerator {
public:
    MSDAOptGenerator() : KernelGenerator("msda_opt") {}

protected:
    // value [B, S, H, D], value_spatial_shapes [L, 2], level_start_index [L],
    // sampling_locations [B, Q, H, L, P, 2], attention_weights [B, Q, H, L, P].
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto value = params.get_input_layout(0).get_shape();
        const auto weights = params.get_input_layout(4).get_shape();
        jit.make("SPATIAL_SIZE", value[1]);
        jit.make("NUM_HEADS", value[2]);
        jit.make("EMBED_DIMS", value[3]);
        jit.make("NUM_QUERY", weights[1]);
        jit.make("NUM_LEVELS", weights[3]);
        jit.make("NUM_POINT", weights[4]);
        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            // One work item per output element (b, q, h, d).
            wgs.global = {1, 1, params.get_output_layout(0).count()};
            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info());
        }};
    }
};

class MSDAOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MSDAOptImpl)

    Stage::Ptr msda = make_stage<MSDAOptGenerator>();

    MSDAOptImpl() : PrimitiveImplOCL(MSDAOpt::get_type_info_static()) {}
    MSDAOptImpl(const program_node& node, const RuntimeParams& params) : MSDAOptImpl() {
        add_stage(msda, params);
    }
    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MSDAOptImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MSDAOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<msda>());
    return std::make_unique<MSDAOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::msda)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MSDAOptImpl)

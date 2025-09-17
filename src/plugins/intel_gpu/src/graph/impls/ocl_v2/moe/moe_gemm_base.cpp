// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "moe_gemm_base.hpp"
#include "moe_gemm_gen_opt.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace ov::intel_gpu::ocl {

JitConstants MoEGemmBase::get_jit_constants(const RuntimeParams& params) const {
    auto jit_constants = KernelGenerator::get_jit_constants(params);
//    auto desc = params.typed_desc<moe_gemm>();

    constexpr size_t subgroup_size = 16;
    jit_constants.make("SUBGROUP_SIZE", subgroup_size);
    return jit_constants;
}

Arguments MoEGemmBase::get_arguments_desc(const RuntimeParams& params) const {
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

    args.push_back({ArgumentDescriptor::Types::INPUT, 0}); // input
    args.push_back({ArgumentDescriptor::Types::INPUT, 1}); // weight
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    args.push_back({ArgumentDescriptor::Types::INPUT, 2}); // input offset
    args.push_back({ArgumentDescriptor::Types::INPUT, 3}); // weight offset
    args.push_back({ArgumentDescriptor::Types::INPUT, 4}); // n_array
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // k
    return args;
}

DispatchDataFunc MoEGemmBase::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        const auto& desc = params.typed_desc<cldnn::moe_gemm>();

        auto& wgs = kd.params.workGroups;
        auto input_layout = params.get_input_layout();
        auto output_layout = params.get_output_layout();

        wgs.global = {1, 1, 1};
        wgs.local = {1, 1, 1};

        auto& scalars = kd.params.scalars;
        scalars.clear();
        scalars.reserve(1);
        ScalarDescriptor s_k{ScalarDescriptor::Types::INT32};
        s_k.v.s32 = 16; // TODO
        scalars.push_back(s_k);
    }};
}

}  // namespace ov::intel_gpu::ocl

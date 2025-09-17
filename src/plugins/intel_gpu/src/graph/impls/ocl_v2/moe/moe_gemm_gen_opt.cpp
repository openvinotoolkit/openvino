// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_gemm_gen_opt.hpp"

#include "../utils/jitter.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "moe_gemm_base.hpp"
#include "moe_gemm.hpp"

namespace ov::intel_gpu::ocl {

JitConstants MoEGemmOptGeneratorBase::get_jit_constants_base(const RuntimeParams& params) const {
    auto jit = MoEGemmBase::get_jit_constants(params);
    std::cout << "TODO : " << __FILE__ << " : " << __LINE__ << std::endl;
    return jit;
}

Arguments MoEGemmOptGeneratorBase::get_arguments_desc_impl(const RuntimeParams& params) const {
    Arguments args;
    std::cout << "TODO : " << __FILE__ << " : " << __LINE__ << std::endl;
    return args;
}

Arguments MoEGemmOptGeneratorMultiToken::get_arguments_desc(const RuntimeParams& params) const {
    return get_arguments_desc_impl(params);
}

JitConstants MoEGemmOptGeneratorMultiToken::get_jit_constants(const RuntimeParams& params) const {
    auto jit = get_jit_constants_base(params);
    std::cout << "TODO : " << __FILE__ << " : " << __LINE__ << std::endl;
    return jit;
}

DispatchDataFunc MoEGemmOptGeneratorMultiToken::get_dispatch_data_func() const {
    std::cout << "TODO : " << __FILE__ << " : " << __LINE__ << std::endl;
    
    return DispatchDataFunc{[](const RuntimeParams& impl_param, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        wgs.global = {1, 1, 1};
        wgs.local = {1, 1, 1};
    }};
}
}  // namespace ov::intel_gpu::ocl
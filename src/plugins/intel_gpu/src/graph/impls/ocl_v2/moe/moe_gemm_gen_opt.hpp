// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_v2/utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"

#include "moe_gemm_inst.h"
#include "moe_gemm_base.hpp"
using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

using RuntimeParams = kernel_impl_params;

class MoEGemmOptGeneratorBase : public MoEGemmBase {
public:
    MoEGemmOptGeneratorBase(std::string_view name, std::string_view stage) : MoEGemmBase(name, stage) {}


    [[nodiscard]] JitConstants get_jit_constants_base(const RuntimeParams& params) const;
    [[nodiscard]] Arguments get_arguments_desc_impl(const RuntimeParams& params) const;

};

class MoEGemmOptGeneratorMultiToken : public MoEGemmOptGeneratorBase {
public:
    explicit MoEGemmOptGeneratorMultiToken() : MoEGemmOptGeneratorBase("moe_gemm_opt", "_prefill") {}

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};
}  // namespace ov::intel_gpu::ocl

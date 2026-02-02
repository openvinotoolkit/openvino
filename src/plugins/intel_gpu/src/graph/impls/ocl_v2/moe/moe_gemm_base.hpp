// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "moe_gemm_inst.h"
#include "openvino/core/type.hpp"

namespace ov::intel_gpu::ocl {
struct MoEGemmRuntimeParams : public ImplRuntimeParams {
    int32_t num_actually_used_experts = 0;
};

struct MoEGemmBase : public KernelGenerator {
    MoEGemmBase(std::string_view name, std::string_view suffix) : KernelGenerator(name, suffix) {}

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override = 0;
    Arguments get_arguments_desc(const RuntimeParams& params) const override = 0;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override = 0;
};
}  // namespace ov::intel_gpu::ocl

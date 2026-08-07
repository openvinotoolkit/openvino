// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace ov::intel_gpu::ocl {

// Correctness-first reference kernel for u2 (2-bit) compressed weights.
// gemmstone micro-kernels and oneDNN have no u2 support; this stage serves all
// token counts when the weight data type is u2.
class GatherMatmulU2RefGenerator : public KernelGenerator {
public:
    explicit GatherMatmulU2RefGenerator() : KernelGenerator("gather_matmul_u2_ref", "_u2_ref") {}

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

}  // namespace ov::intel_gpu::ocl

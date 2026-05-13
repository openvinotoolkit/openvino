// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace ov::intel_gpu::ocl {

class GatherMatmulSortGenerator : public KernelGenerator {
public:
    explicit GatherMatmulSortGenerator() : KernelGenerator("gathermatmul_sort", "_sort") {}

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

}  // namespace ov::intel_gpu::ocl

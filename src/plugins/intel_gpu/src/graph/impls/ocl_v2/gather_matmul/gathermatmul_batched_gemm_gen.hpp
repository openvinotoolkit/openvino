// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "../utils/kernel_generator.hpp"
#include "gather_matmul_gen_micro.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "micro_utils.hpp"

namespace ov::intel_gpu::ocl {

#ifdef ENABLE_ONEDNN_FOR_GPU
// Batched GEMM generator for the prefill path.
// Reuses the same gemmstone micro-kernel as GatherMatmulMicroGenerator but with
// different dispatch (z=groups, y=token_tiles) and scattered output writes.
class GatherMatmulBatchedGemmGenerator : public KernelGenerator {
public:
    explicit GatherMatmulBatchedGemmGenerator() : KernelGenerator("gather_matmul_batched", "_batched_prefill") {}

    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override;
    [[nodiscard]] KernelData get_kernel_data(const kernel_impl_params& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        OPENVINO_THROW("Use overloaded version instead");
    }
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params, const micro::Package& bgm_gemm, const gathermatmul_config& cfg) const;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

    static std::mutex mtx;
};
#endif

}  // namespace ov::intel_gpu::ocl

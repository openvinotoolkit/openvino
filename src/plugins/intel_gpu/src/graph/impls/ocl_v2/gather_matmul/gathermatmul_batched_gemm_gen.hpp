// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "../expert_gemm_gen_utils.hpp"
#include "gather_matmul_gen_micro.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace ov::intel_gpu::ocl {

#ifdef ENABLE_ONEDNN_FOR_GPU
/// Batched GEMM for GatherMatMul: groups with token-scatter write-back.
/// Derives from ExpertGemmBatchedGeneratorBase which provides get_build_options
/// and the gemmstone get_kernel_data template method.
class GatherMatmulBatchedGemmGenerator : public ExpertGemmBatchedGeneratorBase {
public:
    explicit GatherMatmulBatchedGemmGenerator()
        : ExpertGemmBatchedGeneratorBase("gather_matmul_batched", "_batched_prefill") {}

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

    static std::mutex mtx;

protected:
    [[nodiscard]] JitConstants build_jit_constants(const kernel_impl_params& params,
                                                    const micro::Package& bgm_gemm,
                                                    const gathermatmul_config& cfg) const override;
    [[nodiscard]] const char* get_class_name() const override { return "GatherMatmulBatchedGemmGenerator"; }
};
#endif

}  // namespace ov::intel_gpu::ocl

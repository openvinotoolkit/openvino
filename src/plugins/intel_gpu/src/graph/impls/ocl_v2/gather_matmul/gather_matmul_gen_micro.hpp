// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "gather_matmul_inst.h"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gather_matmul.hpp"
#include "micro_utils.hpp"
#include "ocl_v2/utils/jitter.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct GatherMatmulRuntimeParams : public ImplRuntimeParams {
    int32_t n_activated_experts = 0;
    int32_t top_k = 0;
    int32_t n_tokens = 0;
};

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include "micro_utils.hpp"

struct gathermatmul_config {
    bool has_bias = false;
    bool is_weight_quantized = false;
    bool is_weight_symmetric_quantized = false;
    int32_t weight_group_size = -1;
    int32_t weight_scale_idx = -1;
    int32_t weight_zp_idx = -1;
};

class GatherMatmulMicroGenerator : public KernelGenerator {
public:
    explicit GatherMatmulMicroGenerator(bool prefill) : KernelGenerator("gather_matmul", prefill ? "_prefill" : "_generate"), m_is_prefill(prefill) {}

    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override;
    [[nodiscard]] KernelData get_kernel_data(const kernel_impl_params& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        OPENVINO_THROW("Use overloaded version instead");
    }
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params, const micro::Package& bgm_gemm, const gathermatmul_config& cfg) const;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

    static void init_microkernels(const kernel_impl_params& params, micro::Package& gemm_bgm, bool is_prefill);
    static gathermatmul_config get_config(const kernel_impl_params& params);

    bool m_is_prefill;
    static std::mutex mtx;
};
#endif
}  // namespace ov::intel_gpu::ocl

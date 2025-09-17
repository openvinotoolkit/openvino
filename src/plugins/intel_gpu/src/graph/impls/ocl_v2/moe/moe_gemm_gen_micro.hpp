// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "micro_utils.hpp"
#include "ocl_v2/utils/jitter.hpp"

#include "moe_gemm_inst.h"
#include "moe_gemm_gen_opt.hpp"
#include "moe_gemm_base.hpp"
using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {
#ifdef ENABLE_ONEDNN_FOR_GPU
#include "micro_utils.hpp"

struct moe_config {
    bool has_bias = false;
    bool is_activation_quantized = false;
    bool is_activation_symmetric_quantized = false;
    bool is_weight_quantized = false;
    bool is_weight_symmetric_quantized = false;
    int32_t weight_group_size = -1;
};

class MoEGemmMicroGenerator : public MoEGemmOptGeneratorBase {
public:
    explicit MoEGemmMicroGenerator(bool prefill)
        : MoEGemmOptGeneratorBase("moe_gemm", prefill ? "_prefill" : "_generate"),
          m_is_prefill(prefill) {
    }

    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override;

    [[nodiscard]] KernelData get_kernel_data(const kernel_impl_params& params) const override;

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        OPENVINO_THROW("Use overloaded version instead");
    }
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params, const micro::Package& moe_gemm, const moe_config& cfg) const;

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

    static void init_microkernels(const kernel_impl_params& params, micro::Package& gemm_moe, bool is_prefill);

    static moe_config get_moe_cfg(const kernel_impl_params& params);

    bool m_is_prefill;
    static std::mutex mtx;
};
#endif
}  // namespace ov::intel_gpu::ocl

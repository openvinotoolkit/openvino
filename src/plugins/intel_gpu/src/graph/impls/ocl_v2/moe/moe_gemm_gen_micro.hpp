// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "micro_utils.hpp"
#include "moe_gemm_base.hpp"
#include "moe_gemm_gen_opt.hpp"
#include "moe_gemm_inst.h"
#include "ocl_v2/utils/jitter.hpp"
using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {
#ifdef ENABLE_ONEDNN_FOR_GPU
#    include "micro_utils.hpp"

class MoEGemmMicroGenerator : public MoEGemmOptGeneratorBase {
public:
    explicit MoEGemmMicroGenerator(bool prefill) : MoEGemmOptGeneratorBase("moe_gemm", prefill ? "_prefill" : "_generate"), m_is_prefill(prefill) {}

    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override;

    [[nodiscard]] KernelData get_kernel_data(const kernel_impl_params& params) const override;

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        OPENVINO_THROW("Use overloaded version instead");
    }
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params, const micro::Package& moe_gemm, const moe_config& cfg) const;

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

    static void init_microkernels(const kernel_impl_params& params, micro::Package& gemm_moe, bool is_prefill) noexcept;

    bool m_is_prefill;
    static std::mutex mtx;
};
#endif
}  // namespace ov::intel_gpu::ocl

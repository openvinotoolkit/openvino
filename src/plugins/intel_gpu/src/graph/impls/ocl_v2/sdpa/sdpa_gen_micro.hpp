// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "micro_utils.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_base.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

#ifdef ENABLE_ONEDNN_FOR_GPU
class SDPAMicroGenerator : public SDPABase {
public:
    explicit SDPAMicroGenerator(bool prefill) : SDPABase("sdpa_micro", prefill ? "_prefill" : "_generate", false), m_is_prefill(prefill) {}

    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override;
    [[nodiscard]] KernelData get_kernel_data(const kernel_impl_params& params) const override;

    static void update_pa_sdpa_configuration(const sdpa_configuration& sdpa_config);
    size_t get_tile_qsize(const KernelData& kernel_data);

private:
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        OPENVINO_THROW("Use overloaded version instead");
    }
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params, const micro::Package& gemm_kq, const micro::Package& gemm_vs) const;

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

    static void init_microkernels(const kernel_impl_params& params,
                                  const sdpa_configuration& sdpa_config,
                                  micro::Package& gemm_kq,
                                  micro::Package& gemm_vs,
                                  bool is_prefill);
    static void init_sdpa_configuration(const kernel_impl_params& params, sdpa_configuration& config);

    bool m_is_prefill;
    static std::mutex m;

    static constexpr size_t kq_id = 0;
    static constexpr size_t vs_id = 1;
    static constexpr size_t prefill_id = 0;
    static constexpr size_t generate_id = 1;

    static constexpr bool kq_common_scales = false;
    static constexpr bool kq_common_zp = false;
    static constexpr bool vs_common_scales = false;
    static constexpr bool vs_common_zp = false;
};
#endif
}  // namespace ov::intel_gpu::ocl
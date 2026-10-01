// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
// Nothing here uses it, but the files that include this header first rely on it for the oneDNN
// headers to precede intel_gpu/runtime/utils.hpp (see the note at the top of paged_attention_opt.cpp).
#include "micro_utils.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_base.hpp"
#include "sdpa_ocl_hpg.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

#ifdef ENABLE_ONEDNN_FOR_GPU
class SDPAOclGenerator : public SDPABase {
public:
    explicit SDPAOclGenerator(bool prefill) : SDPABase("sdpa_ocl", prefill ? "prefill" : "mixed", false), m_is_prefill(prefill) {}

    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override;

    // Queries per workgroup (the KQ workgroup query tile). Paged attention must use exactly this as the
    // blocked_indexes_start_and_gws_mapping stride, or part of every subsequence is left uncomputed.
    static size_t get_query_block_size(const kernel_impl_params& params);

    // Whether a tiling exists for this (k_head_size, v_head_size) pair: KQ is tiled by k_head_size and the
    // S*V split by v_head_size, so the two need not be equal. Decidable from the descriptor and the arch,
    // because an added stage is compiled even for parameters it is never dispatched with.
    static bool supports_head_sizes(gpu_arch arch, size_t k_head_size, size_t v_head_size);

    // The HpgTier bits this op needs on xe_hpg.
    static uint32_t hpg_tier_required(const kernel_impl_params& params);

    // Whether sdpa_ocl.cl compiles for these layouts: Xe2 or later (xe_hpg only with TEST_USE_SDPA_OCL_HPG=1, and then only
    // for ops whose HpgTier bits are all ready), f16/bf16 Q and output, K/V matching Q or an i8/u4 cache. An added stage
    // is compiled even when it is never dispatched, so everything else must be rejected here.
    static bool supported(const kernel_impl_params& params);

private:
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

    bool m_is_prefill;
};
#endif
}  // namespace ov::intel_gpu::ocl
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <intel_gpu/primitives/paged_attention.hpp>

#include "test_utils.h"
#ifdef ENABLE_ONEDNN_FOR_GPU
#    include "impls/ocl_v2/sdpa/sdpa_ocl_hpg.hpp"
#endif

namespace tests {

// The DPAS kernel plain SDPA and PA PREFILL/MIXED are expected to use on the test device: sdpa_ocl on the
// sdpa_ocl_selected() lane (Xe2+ XMX; xe_hpg only with TEST_USE_SDPA_OCL_HPG=1), none (the opt kernels) otherwise.
// Per-op refusals (head sizes, page layout, token_type_ids, ...) are the caller's to apply.
enum class dpas_backend { none, ocl };

// pure: everything the decision needs is an argument, so a host test can walk arch x env x immad.
// any_hpg_tier_ready: on xe_hpg, whether every HpgTier the op needs is ready (SDPAOclGenerator::supported() refuses the
// op otherwise, and it runs the opt kernels).
inline dpas_backend expected_dpas_backend_for(const cldnn::device_info& info, bool hpg_opt_in, bool any_hpg_tier_ready = true) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    if (!cldnn::paged_attention::sdpa_ocl_selected(info, hpg_opt_in))
        return dpas_backend::none;
    if (info.arch == cldnn::gpu_arch::xe_hpg && !any_hpg_tier_ready)
        return dpas_backend::none;
    return dpas_backend::ocl;
#else
    (void)info;
    (void)hpg_opt_in;
    (void)any_hpg_tier_ready;
    return dpas_backend::none;
#endif
}

// The HpgTier bits a PA op needs on xe_hpg, as SDPAOclGenerator::hpg_tier_required() derives them: `features` = a sink,
// token_type_ids, qq_bias, a sliding window, k_head_size != v_head_size or a runtime scale. 0 without oneDNN.
enum class pa_cache { f16, i8_by_token, i8_by_channel, u4 };
inline uint32_t pa_hpg_tiers(bool features, pa_cache cache = pa_cache::f16) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    using namespace ov::intel_gpu::ocl;
    uint32_t bits = PA_PREFILL | PA_MIXED_F16 | (features ? PA_FEATURES : 0u);
    switch (cache) {
    case pa_cache::f16:
        break;
    case pa_cache::i8_by_token:
        bits |= PA_I8_TOKEN;
        break;
    case pa_cache::i8_by_channel:
        bits |= PA_I8_CHANNEL;
        break;
    case pa_cache::u4:
        bits |= PA_U4;
        break;
    }
    return bits;
#else
    (void)features;
    (void)cache;
    return 0;
#endif
}

// `hpg_tiers`: the HpgTier bits the op under test needs on xe_hpg (SDPAOclGenerator::hpg_tier_required()); 0 = the family's
// base set (plain SDPA: PLAIN_F16_STATIC; PA: PA_PREFILL | PA_MIXED_F16). Ignored off xe_hpg. A caller whose op touches
// more than the base set (a mask, bf16, a dynamic shape, ...) must pass the extra bits, or the helper expects sdpa_ocl for
// an op the tier mask still sends to the opt kernels.
inline dpas_backend expected_dpas_backend(cldnn::engine& engine, bool paged_attention, uint32_t hpg_tiers = 0) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    if (hpg_tiers == 0) {
        hpg_tiers = paged_attention ? (ov::intel_gpu::ocl::PA_PREFILL | ov::intel_gpu::ocl::PA_MIXED_F16) : ov::intel_gpu::ocl::PLAIN_F16_STATIC;
    }
    return expected_dpas_backend_for(engine.get_device_info(),
                                     cldnn::paged_attention::sdpa_ocl_hpg_enabled(),
                                     ov::intel_gpu::ocl::hpg_tiers_cover(hpg_tiers, ov::intel_gpu::ocl::hpg_tiers_ready()));
#else
    (void)engine;
    (void)paged_attention;
    (void)hpg_tiers;
    return dpas_backend::none;
#endif
}

}  // namespace tests

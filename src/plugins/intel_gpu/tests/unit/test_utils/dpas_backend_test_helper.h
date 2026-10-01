// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <intel_gpu/primitives/paged_attention.hpp>

#include "test_utils.h"
#ifdef ENABLE_ONEDNN_FOR_GPU
#    include "impls/ocl_v2/sdpa/sdpa_ocl_hpg.hpp"
#endif

namespace cldnn {
extern bool query_microkernels_supported(cldnn::engine& e, const cldnn::ExecutionConfig& config);
}  // namespace cldnn

namespace tests {

// The DPAS kernel plain SDPA and PA PREFILL/MIXED are expected to use on the test device: sdpa_ocl on the
// sdpa_ocl_selected() lane (Xe2+ XMX; xe_hpg only with TEST_USE_SDPA_OCL_HPG=1; not with TEST_USE_SDPA_OCL=0), sdpa_micro on the other XMX parts with microkernel support, none otherwise.
// Per-op refusals (head sizes, page layout, token_type_ids, ...) are the caller's to apply.
enum class dpas_backend { none, ocl, micro };

// pure: everything the decision needs is an argument, so a host test can walk arch x env x immad x mk.
inline dpas_backend expected_dpas_backend_for(const cldnn::device_info& info,
                                              bool microkernels_supported,
                                              bool ocl_enabled,
                                              bool hpg_opt_in,
                                              bool paged_attention,
                                              size_t k_head_size,
                                              bool any_hpg_tier_ready = true) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    if (!info.supports_immad)
        return dpas_backend::none;
    // TEMP(S9): xe_hpg with no HpgTier ready refuses every op (SDPAOclGenerator::supported()), and the temporary routing
    // sends those to sdpa_micro. The caller states which tiers its op needs (expected_dpas_backend's hpg_tiers) and passes
    // whether they are all ready as any_hpg_tier_ready. Remove with the routing.
    const bool hpg_all_refused = info.arch == cldnn::gpu_arch::xe_hpg && hpg_opt_in && !any_hpg_tier_ready;
    // The sdpa_ocl lane. It still needs the microkernel query today (choose_dpas_backend / supports_micro_sdpa);
    // the default flip drops exactly this term. An ocl branch before the mk check WITHOUT this term would
    // expect ocl on a driver where production returns none.
    if (cldnn::paged_attention::sdpa_ocl_selected(info, ocl_enabled, hpg_opt_in) && !hpg_all_refused)
        return microkernels_supported ? dpas_backend::ocl : dpas_backend::none;
    if (info.arch < cldnn::gpu_arch::xe_hpg || !microkernels_supported)
        return dpas_backend::none;
    // Upstream's xe3p workaround, kept for sdpa_micro only: PA never, plain SDPA not for head sizes <= 64.
    if (info.arch == cldnn::gpu_arch::xe3p && (paged_attention || k_head_size <= 64))
        return dpas_backend::none;
    return dpas_backend::micro;
#else
    (void)info;
    (void)microkernels_supported;
    (void)ocl_enabled;
    (void)hpg_opt_in;
    (void)paged_attention;
    (void)k_head_size;
    (void)any_hpg_tier_ready;
    return dpas_backend::none;
#endif
}

// `hpg_tiers`: the HpgTier bits the op under test needs on xe_hpg (SDPAOclGenerator::hpg_tier_required()); 0 = the family's
// base set (plain SDPA: PLAIN_F16_STATIC; PA: PA_PREFILL | PA_MIXED_F16). Ignored off xe_hpg. A caller whose op touches
// more than the base set (a mask, bf16, a dynamic shape, ...) must pass the extra bits, or the helper expects sdpa_ocl for
// an op the tier mask still sends to sdpa_micro.
inline dpas_backend expected_dpas_backend(cldnn::engine& engine, bool paged_attention, size_t k_head_size, uint32_t hpg_tiers = 0) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    const auto& info = engine.get_device_info();
    // Query only where the old helper did (immad && arch >= xe_hpg): the probe may build a kernel.
    const bool mk = info.supports_immad && info.arch >= cldnn::gpu_arch::xe_hpg &&
                    cldnn::query_microkernels_supported(engine, get_test_default_config(engine));
    if (hpg_tiers == 0) {
        hpg_tiers = paged_attention ? (ov::intel_gpu::ocl::PA_PREFILL | ov::intel_gpu::ocl::PA_MIXED_F16) : ov::intel_gpu::ocl::PLAIN_F16_STATIC;
    }
    return expected_dpas_backend_for(info,
                                     mk,
                                     cldnn::paged_attention::sdpa_ocl_enabled(),
                                     cldnn::paged_attention::sdpa_ocl_hpg_enabled(),
                                     paged_attention,
                                     k_head_size,
                                     ov::intel_gpu::ocl::hpg_tiers_cover(hpg_tiers, ov::intel_gpu::ocl::hpg_tiers_ready()));
#else
    (void)engine;
    (void)paged_attention;
    (void)k_head_size;
    (void)hpg_tiers;
    return dpas_backend::none;
#endif
}

}  // namespace tests

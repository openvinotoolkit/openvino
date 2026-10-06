// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The host-side pieces of the xe_hpg (SG8) sdpa_ocl bring-up that need no kernel_impl_params: the tier mask and the tiling
// limits. Light enough for the unit tests to include.

#pragma once

#include <cstddef>
#include <cstdint>

#include "intel_gpu/runtime/device_info.hpp"

namespace ov::intel_gpu::ocl {

// The op families the xe_hpg (SG8) kernels serve, one bit each. An op needs every bit it touches (hpg_tier_required()) and
// xe_hpg takes the sdpa_ocl lane only when hpg_tiers_ready() has all of them, so an op whose kernel arm does not exist yet is
// refused loudly on the host instead of running the SG16 code on a SG8 device (DG2 drops that DPAS without an error).
// Each step of the xe_hpg plan turns on its own bit in kHpgTiersReady; the whole mask goes away at the default flip.
enum HpgTier : uint32_t {
    PLAIN_F16_STATIC = 1u << 0,  // plain SDPA, f16, static shape, more than one query, no mask/causal/sink/runtime scale
    PLAIN_EXT = 1u << 1,         // plain SDPA with bf16, a mask, causal, a sink, a runtime scale, a dynamic shape or one query
    PLAIN_I8 = 1u << 2,          // plain SDPA on a compressed i8 KV input (u4 is refused outright on xe_hpg, see supported())
    PA_PREFILL = 1u << 3,        // paged attention, both stages compile, so every PA op needs PA_PREFILL and PA_MIXED_F16
    PA_MIXED_F16 = 1u << 4,
    PA_FEATURES = 1u << 5,    // PA with a sink, token_type_ids, qq_bias, a sliding window, k_head_size != v_head_size or a runtime scale
    PA_I8_TOKEN = 1u << 6,    // PA on an i8 BY_TOKEN cache
    PA_I8_CHANNEL = 1u << 7,  // PA on an i8 BY_CHANNEL K cache
    PA_U4 = 1u << 8,          // PA on a u4 cache
};

// Bits ready on xe_hpg. PLAIN_F16_STATIC since plan S6a (the first SG8 kernel arm), PLAIN_EXT since S6b, PLAIN_I8 since S6c; the later steps add theirs.
constexpr uint32_t kHpgTiersReady = PLAIN_F16_STATIC | PLAIN_EXT | PLAIN_I8;

// kHpgTiersReady, or every bit when SDPA_OCL_HPG_TIERS=all (a comma list of tier names selects some). Development only,
// for dumping the jit of every combination: the kernels of a tier that is not ready fail to build. Read once per process.
uint32_t hpg_tiers_ready();
inline bool hpg_tier_ready(HpgTier tier) {
    return (hpg_tiers_ready() & tier) == tier;
}
// Pure: whether `ready` covers every bit of `required` (and `required` is not empty).
inline bool hpg_tiers_cover(uint32_t required, uint32_t ready) {
    return required != 0 && (required & ready) == required;
}

// What a tiling costs on a device: the host-side view of the kernel's local memory and workgroup size.
struct SDPAOclTilingInfo {
    int subgroup_size = 0;
    int sg_per_wg = 0;
    int wg_size = 0;  // sg_per_wg * subgroup_size work-items
    int kq_sg_tile_keys = 0;
    int kq_sg_tile_queries = 0;
    int kq_sg_per_wg_keys = 0;
    int kq_sg_per_wg_queries = 0;
    int sv_sg_tile_values = 0;
    int sv_sg_tile_scores = 0;
    int sv_sg_per_wg_values = 0;
    int sv_sg_per_wg_scores = 0;
    int kq_wg_tile_keys = 0;
    int kq_wg_tile_queries = 0;
    size_t slm_bytes = 0;
};

// The tiling sdpa_ocl uses for these head sizes on `arch`, with the SDPA_OCL_KQ_* overrides applied like choose_config() does;
// false when there is none or it does not fit the device (local memory, workgroup size). Never throws.
bool sdpa_ocl_describe_tiling(cldnn::gpu_arch arch, size_t k_head_size, size_t v_head_size, SDPAOclTilingInfo& info);

// Local memory (bytes) and work-group size sdpa_ocl plans for, per arch (not the queried device, so a dump with a forged arch
// is judged by the arch it stands for).
size_t sdpa_ocl_max_slm_bytes(cldnn::gpu_arch arch);
size_t sdpa_ocl_max_wg_size(cldnn::gpu_arch arch);

}  // namespace ov::intel_gpu::ocl

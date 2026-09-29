// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <intel_gpu/primitives/paged_attention.hpp>

#include "test_utils.h"

namespace cldnn {
extern bool query_microkernels_supported(cldnn::engine& e, const cldnn::ExecutionConfig& config);
}  // namespace cldnn

namespace tests {

// The DPAS kernel plain SDPA and PA PREFILL/MIXED are expected to use on the test device: sdpa_ocl on Xe2+ XMX
// (unless TEST_USE_SDPA_OCL=0), sdpa_micro on the other XMX parts with microkernel support, none otherwise.
// Per-op refusals (head sizes, page layout, token_type_ids, ...) are the caller's to apply.
enum class dpas_backend { none, ocl, micro };

inline dpas_backend expected_dpas_backend(cldnn::engine& engine, bool paged_attention, size_t k_head_size) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    const auto& info = engine.get_device_info();
    if (!info.supports_immad || info.arch < cldnn::gpu_arch::xe_hpg ||
        !cldnn::query_microkernels_supported(engine, get_test_default_config(engine)))
        return dpas_backend::none;
    if (cldnn::paged_attention::sdpa_ocl_selected(info))
        return dpas_backend::ocl;
    // Upstream's xe3p workaround, kept for sdpa_micro only: PA never, plain SDPA not for head sizes <= 64.
    if (info.arch == cldnn::gpu_arch::xe3p && (paged_attention || k_head_size <= 64))
        return dpas_backend::none;
    return dpas_backend::micro;
#else
    (void)engine;
    (void)paged_attention;
    (void)k_head_size;
    return dpas_backend::none;
#endif
}

}  // namespace tests

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <intel_gpu/primitives/paged_attention.hpp>

#include "test_utils.h"

namespace tests {

// The DPAS kernel plain SDPA and PA PREFILL/MIXED are expected to use on the test device: sdpa_ocl on the
// sdpa_ocl_selected() lane (xe_hpc, Xe2+ and xe_hpg XMX) with enable_sdpa_micro on, none (the opt kernels) otherwise.
// Per-op refusals (head sizes, page layout, token_type_ids, ...) are the caller's to apply. GENERATE does not
// follow this: sdpa_ocl_decode ignores enable_sdpa_micro (paged_attention::sdpa_ocl_decode_reader_available()).
enum class dpas_backend { none, ocl };

// pure: everything the decision needs is an argument, so a host test can walk arch x immad. none without oneDNN
// (sdpa_ocl_selected()).
inline dpas_backend expected_dpas_backend_for(const cldnn::device_info& info) {
    return cldnn::paged_attention::sdpa_ocl_selected(info) ? dpas_backend::ocl : dpas_backend::none;
}

// enable_sdpa_micro of the config a test program gets: get_test_default_config(), finalized as the program finalizes
// it, so OV_GPU_ENABLE_SDPA_MICRO=0 reaches it (the env also overrides a value the test sets itself). Off, PREFILL,
// MIXED and plain SDPA take the opt kernels on every device (PagedAttentionOptImpl::choose_sdpa_ocl(),
// SDPAOpt::dpas_lane_op_supported()).
inline bool dpas_kernel_enabled_by_config(cldnn::engine& engine) {
    auto config = get_test_default_config(engine);
    config.finalize(engine);
    return config.get_enable_sdpa_micro();
}

// The device lane (expected_dpas_backend_for()) and the config (dpas_kernel_enabled_by_config()).
inline dpas_backend expected_dpas_backend(cldnn::engine& engine) {
    return dpas_kernel_enabled_by_config(engine) ? expected_dpas_backend_for(engine.get_device_info()) : dpas_backend::none;
}

}  // namespace tests

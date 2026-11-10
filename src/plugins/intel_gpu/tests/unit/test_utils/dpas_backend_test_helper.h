// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <intel_gpu/primitives/paged_attention.hpp>

#include "test_utils.h"

namespace tests {

// The DPAS kernel plain SDPA and PA PREFILL/MIXED are expected to use on the test device: sdpa_ocl on the
// sdpa_ocl_selected() lane (Xe2+ and xe_hpg XMX), none (the opt kernels) otherwise.
// Per-op refusals (head sizes, page layout, token_type_ids, ...) are the caller's to apply.
enum class dpas_backend { none, ocl };

// pure: everything the decision needs is an argument, so a host test can walk arch x immad. none without oneDNN
// (sdpa_ocl_selected()).
inline dpas_backend expected_dpas_backend_for(const cldnn::device_info& info) {
    return cldnn::paged_attention::sdpa_ocl_selected(info) ? dpas_backend::ocl : dpas_backend::none;
}

inline dpas_backend expected_dpas_backend(cldnn::engine& engine) {
    return expected_dpas_backend_for(engine.get_device_info());
}

}  // namespace tests

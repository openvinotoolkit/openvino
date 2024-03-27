// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {

class kv_cache_update_kernel_selector : public kernel_selector_base {
public:
    kv_cache_update_kernel_selector();
    KernelsData GetBestKernels(const Params& params) const override;
    static kv_cache_update_kernel_selector& Instance();
};

class sdpa_kernel_selector : public kernel_selector_base {
public:
    sdpa_kernel_selector();
    KernelsData GetBestKernels(const Params& params) const override;
    static sdpa_kernel_selector& Instance();
};

}  // namespace kernel_selector

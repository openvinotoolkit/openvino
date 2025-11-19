// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {

class reorder_kv_cache_kernel_selector : public kernel_selector_base {
public:
    reorder_kv_cache_kernel_selector();
    KernelsData GetBestKernels(const Params& params) const override;
    static reorder_kv_cache_kernel_selector& Instance();
};

}  // namespace kernel_selector

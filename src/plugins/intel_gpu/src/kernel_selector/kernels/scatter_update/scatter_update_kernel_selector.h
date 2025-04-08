// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class scatter_update_kernel_selector : public kernel_selector_base {
public:
    static scatter_update_kernel_selector& Instance() {
        static scatter_update_kernel_selector instance_;
        return instance_;
    }

    scatter_update_kernel_selector();

    virtual ~scatter_update_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

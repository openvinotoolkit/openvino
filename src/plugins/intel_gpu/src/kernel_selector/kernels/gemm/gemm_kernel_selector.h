// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class gemm_kernel_selector : public kernel_selector_base {
public:
    static gemm_kernel_selector& Instance() {
        static gemm_kernel_selector instance;
        return instance;
    }

    gemm_kernel_selector();
    virtual ~gemm_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

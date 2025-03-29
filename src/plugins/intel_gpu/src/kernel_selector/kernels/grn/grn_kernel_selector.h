// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class grn_kernel_selector : public kernel_selector_base {
public:
    static grn_kernel_selector& Instance() {
        static grn_kernel_selector instance_;
        return instance_;
    }

    grn_kernel_selector();
    virtual ~grn_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

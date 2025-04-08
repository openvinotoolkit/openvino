// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class activation_kernel_selector : public kernel_selector_base {
public:
    static activation_kernel_selector& Instance() {
        static activation_kernel_selector instance_;
        return instance_;
    }

    activation_kernel_selector();

    virtual ~activation_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

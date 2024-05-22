// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class sdpa_kernel_selector : public kernel_selector_base {
public:
    static sdpa_kernel_selector& Instance() {
        static sdpa_kernel_selector instance_;
        return instance_;
    }

    sdpa_kernel_selector();

    virtual ~sdpa_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

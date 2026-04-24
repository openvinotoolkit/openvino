// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {

class scaled_shifted_clamp_experimental_kernel_selector : public kernel_selector_base {
public:
    static scaled_shifted_clamp_experimental_kernel_selector& Instance() {
        static scaled_shifted_clamp_experimental_kernel_selector instance_;
        return instance_;
    }

    scaled_shifted_clamp_experimental_kernel_selector();
    ~scaled_shifted_clamp_experimental_kernel_selector() override = default;

    KernelsData GetBestKernels(const Params& params) const override;
};

}  // namespace kernel_selector

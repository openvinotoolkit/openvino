// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class cum_sum_kernel_selector : public kernel_selector_base {
public:
    static cum_sum_kernel_selector& Instance() {
        static cum_sum_kernel_selector instance_;
        return instance_;
    }

    cum_sum_kernel_selector();
    virtual ~cum_sum_kernel_selector() = default;

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

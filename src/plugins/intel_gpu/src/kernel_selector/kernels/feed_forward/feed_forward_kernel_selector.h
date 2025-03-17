// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class feed_forward_kernel_selector : public kernel_selector_base {
public:
    static feed_forward_kernel_selector& Instance() {
        static feed_forward_kernel_selector instance_;
        return instance_;
    }

    feed_forward_kernel_selector();

    virtual ~feed_forward_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

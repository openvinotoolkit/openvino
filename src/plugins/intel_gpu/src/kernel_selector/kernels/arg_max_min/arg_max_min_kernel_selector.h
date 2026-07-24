// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class arg_max_min_kernel_selector : public kernel_selector_base {
public:
    static arg_max_min_kernel_selector& Instance() {
        static arg_max_min_kernel_selector instance_;
        return instance_;
    }

    arg_max_min_kernel_selector();

    ~arg_max_min_kernel_selector() override {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

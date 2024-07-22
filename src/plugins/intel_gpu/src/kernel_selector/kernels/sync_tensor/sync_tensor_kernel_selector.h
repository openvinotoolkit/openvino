// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class sync_tensor_kernel_selector : public kernel_selector_base {
public:
    static sync_tensor_kernel_selector& Instance() {
        static sync_tensor_kernel_selector instance_;
        return instance_;
    }

    sync_tensor_kernel_selector();

    virtual ~sync_tensor_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

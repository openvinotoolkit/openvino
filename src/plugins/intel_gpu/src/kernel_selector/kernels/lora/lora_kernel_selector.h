// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class lora_kernel_selector : public kernel_selector_base {
public:
    static lora_kernel_selector& Instance() {
        static lora_kernel_selector instance_;
        return instance_;
    }

    lora_kernel_selector();

    virtual ~lora_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

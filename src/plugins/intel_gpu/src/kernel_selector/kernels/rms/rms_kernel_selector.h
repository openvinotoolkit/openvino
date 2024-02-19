// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class rms_kernel_selector : public kernel_selector_base {
public:
    static rms_kernel_selector& Instance() {
        static rms_kernel_selector instance_;
        return instance_;
    }

    rms_kernel_selector();

    virtual ~rms_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

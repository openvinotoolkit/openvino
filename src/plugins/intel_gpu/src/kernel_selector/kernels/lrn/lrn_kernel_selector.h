﻿// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class lrn_kernel_selector : public kernel_selector_base {
public:
    static lrn_kernel_selector& Instance() {
        static lrn_kernel_selector instance_;
        return instance_;
    }

    lrn_kernel_selector();

    virtual ~lrn_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

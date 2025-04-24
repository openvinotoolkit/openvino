// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {

class unique_count_kernel_selector : public kernel_selector_base {
public:
    unique_count_kernel_selector();
    KernelsData GetBestKernels(const Params& params) const override;
    static unique_count_kernel_selector& Instance();
};

class unique_gather_kernel_selector : public kernel_selector_base {
public:
    unique_gather_kernel_selector();
    KernelsData GetBestKernels(const Params& params) const override;
    static unique_gather_kernel_selector& Instance();
};

}  // namespace kernel_selector

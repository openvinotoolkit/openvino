// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {

/*
 * GridSample kernel selector.
 */
class grid_sample_kernel_selector : public kernel_selector_base {
public:
    grid_sample_kernel_selector();
    KernelsData GetBestKernels(const Params& params) const override;
    static grid_sample_kernel_selector& Instance();
};

}  // namespace kernel_selector

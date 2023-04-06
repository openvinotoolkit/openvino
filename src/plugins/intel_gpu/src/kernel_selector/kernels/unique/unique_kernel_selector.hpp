// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {

/*
 * Unique kernel selector.
 */
class unique_kernel_selector : public kernel_selector_base {
public:
    unique_kernel_selector();
    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;
    static unique_kernel_selector& Instance();
};

class unique_reshape_kernel_selector : public kernel_selector_base {
public:
    unique_reshape_kernel_selector();
    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;
    static unique_reshape_kernel_selector& Instance();
};

}  // namespace kernel_selector

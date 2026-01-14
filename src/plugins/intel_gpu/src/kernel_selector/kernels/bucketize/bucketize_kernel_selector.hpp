// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "kernel_selector.h"

namespace kernel_selector {

/*
 * Bucketize kernel selector.
 */
class bucketize_kernel_selector : public kernel_selector_base {
public:
    bucketize_kernel_selector();
    KernelsData GetBestKernels(const Params& params) const override;
    static bucketize_kernel_selector& Instance();
};

}  // namespace kernel_selector

// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class bucketize_kernel_selector : public kernel_selector_base {
public:
    static bucketize_kernel_selector& Instance() {
        static bucketize_kernel_selector instance_;
        return instance_;
    }

    bucketize_kernel_selector();

    virtual ~bucketize_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class shape_of_kernel_selector : public kernel_selector_base {
public:
    static shape_of_kernel_selector& Instance() {
        static shape_of_kernel_selector instance_;
        return instance_;
    }

    shape_of_kernel_selector();

    virtual ~shape_of_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

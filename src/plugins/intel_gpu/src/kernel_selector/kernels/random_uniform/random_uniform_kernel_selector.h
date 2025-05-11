// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_selector.h"

#pragma once

namespace kernel_selector {

/**
 * GPU kernel selector for the RandomUniform-8 operation
 */
class random_uniform_kernel_selector: public kernel_selector_base {
public:
    static random_uniform_kernel_selector& Instance() {
        static random_uniform_kernel_selector instance_;
        return instance_;
    }

    KernelsData GetBestKernels(const Params &params) const override;

private:
    random_uniform_kernel_selector();
};

}  // namespace kernel_selector

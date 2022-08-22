// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_selector.h"

#pragma once

namespace kernel_selector {

/**
 * GPU kernel selector for the PriorBox operation
 */
class prior_box_kernel_selector : public kernel_selector_base {
public:
    static prior_box_kernel_selector& Instance() {
        static prior_box_kernel_selector instance_;
        return instance_;
    }

    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;

private:
    prior_box_kernel_selector();
};

}  // namespace kernel_selector

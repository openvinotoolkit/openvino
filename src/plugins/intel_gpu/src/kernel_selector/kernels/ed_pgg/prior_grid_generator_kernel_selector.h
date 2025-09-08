// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <kernel_selector.h>

namespace kernel_selector {
class experimental_detectron_prior_grid_generator_kernel_selector : public kernel_selector_base {
public:
    static experimental_detectron_prior_grid_generator_kernel_selector& Instance() {
        static experimental_detectron_prior_grid_generator_kernel_selector instance_;
        return instance_;
    }

    experimental_detectron_prior_grid_generator_kernel_selector();

    virtual ~experimental_detectron_prior_grid_generator_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

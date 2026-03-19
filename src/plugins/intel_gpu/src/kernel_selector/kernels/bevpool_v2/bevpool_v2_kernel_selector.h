// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {

class bevpool_v2_kernel_selector : public kernel_selector_base {
public:
    static bevpool_v2_kernel_selector& Instance() {
        static bevpool_v2_kernel_selector instance;
        return instance;
    }

    bevpool_v2_kernel_selector();

    KernelsData GetBestKernels(const Params& params) const override;
};

}  // namespace kernel_selector

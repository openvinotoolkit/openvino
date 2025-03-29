// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class convert_color_kernel_selector : public kernel_selector_base {
public:
    static convert_color_kernel_selector& Instance() {
        static convert_color_kernel_selector instance_;
        return instance_;
    }

    convert_color_kernel_selector();

    virtual ~convert_color_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

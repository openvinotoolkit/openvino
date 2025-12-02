// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class sparse_fill_empty_rows_kernel_selector : public kernel_selector_base {
public:
    static sparse_fill_empty_rows_kernel_selector& Instance() {
        static sparse_fill_empty_rows_kernel_selector instance_;
        return instance_;
    }

    sparse_fill_empty_rows_kernel_selector();

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

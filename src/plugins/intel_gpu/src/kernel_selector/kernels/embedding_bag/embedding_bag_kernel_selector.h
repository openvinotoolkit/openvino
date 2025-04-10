// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class embedding_bag_kernel_selector : public kernel_selector_base {
public:
    static embedding_bag_kernel_selector& Instance() {
        static embedding_bag_kernel_selector instance_;
        return instance_;
    }

    embedding_bag_kernel_selector();

    virtual ~embedding_bag_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

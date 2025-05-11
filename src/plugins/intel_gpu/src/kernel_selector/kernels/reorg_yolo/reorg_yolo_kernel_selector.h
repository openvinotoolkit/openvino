// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class reorg_yolo_kernel_selector : public kernel_selector_base {
public:
    static reorg_yolo_kernel_selector& Instance() {
        static reorg_yolo_kernel_selector instance_;
        return instance_;
    }

    reorg_yolo_kernel_selector();

    virtual ~reorg_yolo_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

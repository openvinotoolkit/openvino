// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class STFT_kernel_selector : public kernel_selector_base {
public:
    static STFT_kernel_selector& Instance() {
        static STFT_kernel_selector instance;
        return instance;
    }

    STFT_kernel_selector();

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector

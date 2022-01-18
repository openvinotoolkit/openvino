// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class ReorderWeightsKernelSelctor : public kernel_selector_base {
public:
    static ReorderWeightsKernelSelctor& Instance() {
        static ReorderWeightsKernelSelctor instance_;
        return instance_;
    }

    ReorderWeightsKernelSelctor();

    virtual ~ReorderWeightsKernelSelctor() {}

    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector
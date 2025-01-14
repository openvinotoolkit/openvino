// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../fully_connected_params.h"
#include "kernel_base_cm.h"

namespace kernel_selector {
class FullyConnected_cm_example : public KernelBaseCM {
public:
    FullyConnected_cm_example() : KernelBaseCM("fully_connected_example") {}
    virtual ~FullyConnected_cm_example() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p) const override;
};
}  // namespace kernel_selector

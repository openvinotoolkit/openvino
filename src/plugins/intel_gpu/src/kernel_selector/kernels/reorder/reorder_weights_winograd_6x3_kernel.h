// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderWeightsWinograd6x3Kernel : public ReorderKernelBase {
public:
    ReorderWeightsWinograd6x3Kernel() : ReorderKernelBase("reorder_weights_winograd_6x3_s1") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const reorder_weights_params& arg) const override;
};
}  // namespace kernel_selector

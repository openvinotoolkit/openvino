// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderWeightsImage_fyx_b_Kernel : public ReorderKernelBase {
public:
    ReorderWeightsImage_fyx_b_Kernel() : ReorderKernelBase("reorder_weights_image_2d_c4_fyx_b") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const reorder_weights_params& arg) const override;
};
}  // namespace kernel_selector

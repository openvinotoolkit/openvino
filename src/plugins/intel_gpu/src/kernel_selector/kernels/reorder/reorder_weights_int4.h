// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderWeightsKernelInt4 : public ReorderKernelBase {
public:
    ReorderWeightsKernelInt4() : ReorderKernelBase("reorder_weights_int4") {}
    virtual ~ReorderWeightsKernelInt4() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const reorder_weights_params& arg) const override;

protected:
    bool Validate(const Params& params) const override;
};
}  // namespace kernel_selector

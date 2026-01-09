// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderWeightsKernelInt2 : public ReorderKernelBase {
public:
    ReorderWeightsKernelInt2() : ReorderKernelBase("reorder_weights_int2") {}
    virtual ~ReorderWeightsKernelInt2() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const reorder_weights_params& arg) const override;

protected:
    JitConstants GetJitConstants(const reorder_weights_params& params) const override;
    bool Validate(const Params& params) const override;
};
}  // namespace kernel_selector

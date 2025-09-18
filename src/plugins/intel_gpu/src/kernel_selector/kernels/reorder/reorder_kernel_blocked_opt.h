// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderKernelBlockedOpt : public ReorderKernelBase {
public:
    ReorderKernelBlockedOpt() : ReorderKernelBase("reorder_data_blocked_opt") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    JitConstants GetJitConstants(const reorder_params& params) const override;

protected:
    bool Validate(const Params& p) const override;
    DispatchData SetDefault(const reorder_params& arg) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

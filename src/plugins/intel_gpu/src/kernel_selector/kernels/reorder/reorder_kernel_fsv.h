// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderKernel_fsv : public ReorderKernelBase {
public:
    ReorderKernel_fsv() : ReorderKernelBase("reorder_data_fsv") {}

    bool Validate(const Params& p) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
protected:
    JitConstants GetJitConstants(const reorder_params& params) const override;
    CommonDispatchData SetDefault(const reorder_params& params) const override;
};
}  // namespace kernel_selector

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cum_sum_kernel_base.h"

namespace kernel_selector {
class CumSumKernelRef : public CumSumKernelBase {
public:
    CumSumKernelRef() : CumSumKernelBase("cum_sum_ref") {}
    virtual ~CumSumKernelRef() = default;
protected:
    JitConstants GetJitConstants(const cum_sum_params& params, DispatchData dispatchData) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector

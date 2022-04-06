// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cum_sum_kernel_base.h"

namespace kernel_selector {
class CumSumKernelPartialSum : public CumSumKernelBase {
public:
    CumSumKernelPartialSum() : CumSumKernelBase("cum_sum_partial_sum") {}
    virtual ~CumSumKernelPartialSum() = default;

    ParamsKey GetSupportedKey() const override;
protected:
    struct MultiDispatchData {
        DispatchData stage_1;
        DispatchData stage_final;
    };

    JitConstants GetJitConstants(const cum_sum_params& params, DispatchData dispatchData) const override;
    KernelsData GetMultiStageKernelsData(const Params& params, const optional_params&) const;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    MultiDispatchData SetDefaultForMulti(const cum_sum_params& params) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector

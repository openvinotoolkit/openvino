// Copyright (C) 2018-2025 Intel Corporation
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
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
protected:
    struct MultiDispatchData {
        DispatchData stage_1;
        DispatchData stage_final;
    };

    JitConstants GetJitConstants(const cum_sum_params& params, DispatchData dispatchData) const override;
    KernelsData GetMultiStageKernelsData(const Params& params) const;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    MultiDispatchData SetDefaultForMulti(const cum_sum_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
};
}  // namespace kernel_selector

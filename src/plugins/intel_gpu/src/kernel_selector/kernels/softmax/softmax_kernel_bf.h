// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "softmax_kernel_base.h"

namespace kernel_selector {
class SoftmaxKernel_bf : public SoftmaxKernelBaseBF {
public:
    using Parent = SoftmaxKernelBaseBF;
    SoftmaxKernel_bf() : Parent("softmax_gpu_bf") {}
    virtual ~SoftmaxKernel_bf() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    DispatchData SetDefault(const softmax_params& params) const override;
    JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
    std::vector<KernelBase::FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE };
    }
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

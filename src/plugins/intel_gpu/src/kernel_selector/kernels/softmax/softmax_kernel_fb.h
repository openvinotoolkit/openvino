// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "softmax_kernel_base.h"

namespace kernel_selector {
class SoftmaxKernel_fb : public SoftmaxKernelBaseBF {
public:
    using Parent = SoftmaxKernelBaseBF;
    SoftmaxKernel_fb() : SoftmaxKernelBaseBF("softmax_gpu_fb") {}
    virtual ~SoftmaxKernel_fb() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p) const override;
    DispatchData SetDefault(const softmax_params& params) const override;
    JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const override;
    std::vector<KernelBase::FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE };
    }
};
}  // namespace kernel_selector

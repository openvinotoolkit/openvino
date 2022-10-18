// Copyright (C) 2018-2022 Intel Corporation
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

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    DispatchData SetDefault(const softmax_params& params, const optional_params& optParams) const override;
    JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const override;
    std::vector<KernelBase::FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE };
    }
};
}  // namespace kernel_selector

// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "softmax_kernel_base.h"

namespace kernel_selector {
class SoftmaxKernelBlocked : public SoftmaxKernelBase {
public:
    using Parent = SoftmaxKernelBase;
    SoftmaxKernelBlocked() : Parent("softmax_gpu_blocked") {}
    virtual ~SoftmaxKernelBlocked() = default;

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const override;
    DispatchData SetDefault(const softmax_params& params, const optional_params& optParams) const override;
};
}  // namespace kernel_selector

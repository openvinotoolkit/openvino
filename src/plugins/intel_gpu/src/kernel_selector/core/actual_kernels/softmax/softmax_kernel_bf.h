// Copyright (C) 2018-2022 Intel Corporation
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

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const softmax_params& params, const optional_params& optParams) const override;
};
}  // namespace kernel_selector
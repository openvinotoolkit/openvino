// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "softmax_items_class_kernel_base.h"

namespace kernel_selector {
class SoftmaxKernelRef : public SoftmaxItemsClassKernelBase {
public:
    using Parent = SoftmaxItemsClassKernelBase;
    SoftmaxKernelRef() : Parent("softmax_gpu_ref") {}
    virtual ~SoftmaxKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    DispatchData SetDefault(const softmax_params& params) const override;
    JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

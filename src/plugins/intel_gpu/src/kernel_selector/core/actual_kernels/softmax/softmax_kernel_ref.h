// Copyright (C) 2018-2022 Intel Corporation
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

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    DispatchData SetDefault(const softmax_params& params, const optional_params& optParams) const override;
};
}  // namespace kernel_selector
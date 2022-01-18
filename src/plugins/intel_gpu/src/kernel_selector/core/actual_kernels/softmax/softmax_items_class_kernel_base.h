// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "softmax_kernel_base.h"
#include <vector>

namespace kernel_selector {
class SoftmaxItemsClassKernelBase : public SoftmaxKernelBase {
public:
    using SoftmaxKernelBase::SoftmaxKernelBase;
    virtual ~SoftmaxItemsClassKernelBase() {}

protected:
    JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const override;
    static ParamsKey GetDefaultSupportedKey();
    static std::vector<size_t> GetSoftmaxDimGlobalSizes(SoftmaxDim dim, const DataTensor& output);
};
}  // namespace kernel_selector

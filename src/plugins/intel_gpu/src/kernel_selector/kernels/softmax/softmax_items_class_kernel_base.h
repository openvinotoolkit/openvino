// Copyright (C) 2018-2025 Intel Corporation
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
    Datatype GetAccumulatorType(const softmax_params& params) const {
        if (params.inputs[0].GetDType() == Datatype::F16)
            return Datatype::F16;
        else
            return Datatype::F32;
    }
    std::vector<KernelBase::FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE };
    }
};
}  // namespace kernel_selector

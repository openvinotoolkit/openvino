// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SoftMaxParams
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct softmax_params : public base_params {
    softmax_params() : base_params(KernelType::SOFT_MAX) {}

    SoftmaxDim dim = SoftmaxDim::FEATURE;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        k.EnableSoftmaxDim(dim);
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SoftmaxKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SoftmaxKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~SoftmaxKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        size_t itemsNum;
        size_t leftovers;
        size_t dataSetsCount;
        size_t dataSetSize;
        size_t normIndex;  // which dimension (from in-memory representation) is normalized, e.g. for bfyx and
                           // softmax::normalize_f, it will be f's index == 2 (used only by naive kernel)
        size_t subgroupBlockSize;
    };

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const softmax_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    Datatype GetActivationType(const softmax_params& params) const {
        if (params.inputs[0].GetDType() == Datatype::F16)
            return Datatype::F16;
        else
            return Datatype::F32;
    }
};

class SoftmaxKernelBaseBF : public SoftmaxKernelBase {
public:
    using Parent = SoftmaxKernelBase;
    using Parent::Parent;
    virtual ~SoftmaxKernelBaseBF() {}

protected:
    bool Validate(const Params&) const override;
    DispatchData SetDefault(const softmax_params& params) const override;
};
}  // namespace kernel_selector

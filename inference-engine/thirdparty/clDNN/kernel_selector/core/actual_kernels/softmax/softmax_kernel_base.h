// Copyright (C) 2018-2021 Intel Corporation
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
// softmax_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct softmax_optional_params : optional_params {
    softmax_optional_params() : optional_params(KernelType::SOFT_MAX) {}
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
    };

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const softmax_params& params, const optional_params& optParams) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params& optParams) const;
};

class SoftmaxKernelBaseBF : public SoftmaxKernelBase {
public:
    using Parent = SoftmaxKernelBase;
    using Parent::Parent;
    virtual ~SoftmaxKernelBaseBF() {}

protected:
    bool Validate(const Params&, const optional_params&) const override;
    DispatchData SetDefault(const softmax_params& params, const optional_params& optParams) const override;
};
}  // namespace kernel_selector

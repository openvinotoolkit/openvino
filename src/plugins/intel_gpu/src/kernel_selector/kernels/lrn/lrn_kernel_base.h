// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lrn_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lrn_params : public base_params {
    lrn_params() : base_params(KernelType::LRN) {}

    LRNMode normMode = LRNMode::ACROSS_CHANNEL;
    KernelDividerMode divMode = KernelDividerMode::DONT_CARE;
    float alpha = 0.f;
    float beta = 0.f;
    float k = 0.f;
    uint32_t localSize = 0;

    ParamsKey GetParamsKey() const override {
        ParamsKey _k = base_params::GetParamsKey();

        _k.EnableLRNMode(normMode);
        _k.EnableLRNKernelDividerMode(divMode);

        return _k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lrn_kernel_base
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class LRNKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~LRNKernelBase() {}

    using DispatchData = CommonDispatchData;

protected:
    bool Validate(const Params& p) const override;
    virtual JitConstants GetJitConstants(const lrn_params& params, const DispatchData& dispatchData) const;
    virtual DispatchData SetDefault(const lrn_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
};
}  // namespace kernel_selector

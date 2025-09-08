// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// select_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct select_params : public base_params {
    select_params() : base_params(KernelType::SELECT) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SelectKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SelectKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~SelectKernelBase() {}

    using DispatchData = CommonDispatchData;
    JitConstants GetJitConstantsCommon(const select_params& params) const;

protected:
    bool Validate(const Params& p) const override;
    virtual JitConstants GetJitConstants(const select_params& params) const;
    virtual DispatchData SetDefault(const select_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

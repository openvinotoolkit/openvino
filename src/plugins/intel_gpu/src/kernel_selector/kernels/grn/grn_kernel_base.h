// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <string>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// grn_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct grn_params : public base_params {
    grn_params() : base_params(KernelType::GRN) {}

    float bias = 1.0f;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GRNKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class GRNKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~GRNKernelBase() {}
    using DispatchData = CommonDispatchData;

protected:
    virtual JitConstants GetJitConstants(const grn_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const grn_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
};
}  // namespace kernel_selector

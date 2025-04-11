// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ISTFT
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ISTFT_params : public base_params {
    ISTFT_params() : base_params(KernelType::ISTFT), center(false), normalized(false) {}
    bool center;
    bool normalized;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ISTFTKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ISTFTKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const ISTFT_params& params) const;
    static DispatchData SetDefault(const ISTFT_params& params);
    KernelsData GetCommonKernelsData(const Params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

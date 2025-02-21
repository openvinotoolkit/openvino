// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// STFT
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct STFT_params : public base_params {
    STFT_params() : base_params(KernelType::STFT), transpose_frames(false) {}
    bool transpose_frames;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// STFTKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class STFTKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

protected:
    virtual JitConstants GetJitConstants(const STFT_params& params) const;
    virtual CommonDispatchData CalcLaunchConfig(const STFT_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    bool Validate(const Params& p) const override;
};
}  // namespace kernel_selector

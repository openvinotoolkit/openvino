// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// swiglu_params
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct swiglu_params : public base_params {
    swiglu_params() : base_params(KernelType::SWIGLU), axis(0), split_length(0) {}
    int32_t axis;
    int32_t split_length;
};

class SwiGLUKernelRef : public KernelBaseOpenCL {
public:
    SwiGLUKernelRef() : KernelBaseOpenCL("swiglu_gpu_ref") {}
    virtual ~SwiGLUKernelRef() {}

    virtual JitConstants GetJitConstants(const swiglu_params& params) const;
    virtual CommonDispatchData SetDefault(const swiglu_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    Datatype GetAccumulatorType(const swiglu_params& params) const;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params&) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

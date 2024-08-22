// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include "intel_gpu/op/swiglu.hpp"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// swiglu_params
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct swiglu_params : public base_params {
    swiglu_params() : base_params(KernelType::SWIGLU), axis(0), split_length(0),
    glu_type(ov::intel_gpu::op::SwiGLU::GluType::Swish), split_to_glu_idx(0) {}
    int32_t axis;
    int32_t split_length;
    ov::intel_gpu::op::SwiGLU::GluType glu_type;
    int32_t split_to_glu_idx;
};

class SwiGLUKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~SwiGLUKernelBase() {}

    virtual JitConstants GetJitConstants(const swiglu_params& params, const CommonDispatchData& dispatchData) const;
    virtual CommonDispatchData SetDefault(const swiglu_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
    Datatype GetAccumulatorType(const swiglu_params& params) const;

protected:
    bool Validate(const Params&) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

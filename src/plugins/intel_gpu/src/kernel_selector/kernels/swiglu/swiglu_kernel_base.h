// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include "ov_ops/glu.hpp"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// swiglu_params
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct swiglu_params : public base_params {
    swiglu_params() : base_params(KernelType::SWIGLU), axis(0), split_length(0),
    glu_type(ov::op::internal::GLU::GluType::Swish), split_to_glu_idx(0) {}
    int32_t axis;
    int32_t split_length;
    ov::op::internal::GLU::GluType glu_type;
    int32_t split_to_glu_idx;
};

struct swiglu_fuse_params : fuse_params {
    explicit swiglu_fuse_params(int32_t axis, size_t split_lengths, size_t split_to_glu_idx)
        : fuse_params(KernelType::SWIGLU),
            axis(axis),
            split_length(split_lengths),
            split_to_glu_idx(split_to_glu_idx) {}
    int32_t axis;
    size_t split_length;
    size_t split_to_glu_idx;
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

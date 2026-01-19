// Copyright (C) 2018-2026 Intel Corporation
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
    swiglu_params()
        : base_params(KernelType::SWIGLU),
          axis(0),
          glu_stride(0),
          glu_type(ov::op::internal::GLU::GluType::Swish),
          gate_idx(0),
          clamp_min(std::numeric_limits<float>::lowest()),
          clamp_max(std::numeric_limits<float>::max()),
          swish_beta(1.0f),
          up_add_val(0.0f) {}
    int32_t axis;
    int32_t glu_stride;
    ov::op::internal::GLU::GluType glu_type;
    int32_t gate_idx;
    float clamp_min;
    float clamp_max;
    float swish_beta;
    float up_add_val;
};

struct swiglu_fuse_params : fuse_params {
    explicit swiglu_fuse_params(int32_t axis, size_t glu_stride, size_t gate_idx, size_t clamp_min, size_t clamp_max, float swish_beta, float up_add_val)
        : fuse_params(KernelType::SWIGLU),
          axis(axis),
          glu_stride(glu_stride),
          gate_idx(gate_idx),
          clamp_min(clamp_min),
          clamp_max(clamp_max),
          swish_beta(swish_beta),
          up_add_val(up_add_val) {}
    int32_t axis;
    size_t glu_stride;
    size_t gate_idx;
    float clamp_min;
    float clamp_max;
    float swish_beta;
    float up_add_val;
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

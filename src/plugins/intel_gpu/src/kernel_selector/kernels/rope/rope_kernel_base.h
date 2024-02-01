// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// rope_params
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct rope_params : public base_params {
    rope_params() : base_params(KernelType::ROPE) {}
    size_t head_cnt;
    size_t head_size;
    size_t rotary_ndims;

    size_t slice_start;
    size_t slice_stop;
    size_t axis;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// rope_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct rope_optional_params : optional_params {
    rope_optional_params() : optional_params(KernelType::ROPE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// RoPEKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class RoPEKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~RoPEKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const rope_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const rope_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

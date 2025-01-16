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
    size_t head_cnt = 0;
    size_t head_size = 0;
    size_t rotary_ndims = 0;

    size_t slice_start = 0;
    size_t slice_stop = 0;
    size_t axis = 0;
    size_t num_of_inputs = 0;
    size_t gather_rank = 0;

    bool is_qwen = false;
    bool is_chatglm = false;
    bool support_2d_rope = false;
    bool transposed_input = false;
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
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const rope_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const rope_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

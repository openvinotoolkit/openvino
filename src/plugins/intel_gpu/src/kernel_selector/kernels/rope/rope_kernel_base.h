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
    size_t num_of_inputs;
    size_t gather_rank;

    bool is_qwen;
    bool is_chatglm;
    bool transposed_input;
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

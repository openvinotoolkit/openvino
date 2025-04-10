// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// border_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct border_params : public base_params {
    DimTensor<int32_t> lt_sizes;
    DimTensor<int32_t> rb_sizes;
    BorderType b_type;
    float border_value;
    bool allow_negative_pad;

    ArgType begin_type;
    ArgType end_type;
    ArgType pad_value_type;


    border_params() : base_params(KernelType::BORDER), b_type(BorderType::CONSTANT),
                      border_value(0.0f), allow_negative_pad(false),
                      begin_type(ArgType::Constant), end_type(ArgType::Constant), pad_value_type(ArgType::Constant) {}

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();
        // k.EnableBorderType(b_type);
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BorderKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BorderKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const border_params& params) const;
    DispatchData SetDefault(const border_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    bool SkipKernelExecution(const border_params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

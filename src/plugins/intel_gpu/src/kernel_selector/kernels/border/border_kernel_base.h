// Copyright (C) 2018-2023 Intel Corporation
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
    DimTensor<> lt_sizes;
    DimTensor<> rb_sizes;
    BorderType b_type;
    float border_value;

    ArgType begin_type;
    ArgType end_type;
    ArgType pad_value_type;

    border_params() : base_params(KernelType::BORDER), b_type(BorderType::CONSTANT), border_value(0.0f) {}

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();
        // k.EnableBorderType(b_type);
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// border_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct border_optional_params : optional_params {
    border_optional_params() : optional_params(KernelType::BORDER) {}
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
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
};
}  // namespace kernel_selector

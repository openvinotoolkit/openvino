// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// arg_max_min_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct arg_max_min_params : public base_params {
    arg_max_min_params() : base_params(KernelType::ARG_MAX_MIN) {}

    ArgMaxMinAxis argMaxMinAxis;
    ArgMaxMinOut argMaxMinOut = ArgMaxMinOut::MAX;
    ArgMaxMinSortType argMaxMinSortType = ArgMaxMinSortType::VALUE;
    uint32_t topK = 1;
    uint32_t outputs_num = 1;
    bool values_first = false;
    bool stable = false;

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();
        k.EnableArgMaxMinAxis(argMaxMinAxis);

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ArgMaxMinKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ArgMaxMinKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~ArgMaxMinKernelBase() {}

    struct DispatchData : public CommonDispatchData {
    };

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const arg_max_min_params& params) const;
    virtual DispatchData SetDefault(const arg_max_min_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

// Copyright (C) 2018-2023 Intel Corporation
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
    bool has_second_output = false;
    bool use_multiple_outputs = false;
    bool stable = false;

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();
        k.EnableArgMaxMinAxis(argMaxMinAxis);

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// arg_max_min_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct arg_max_min_optional_params : optional_params {
    arg_max_min_optional_params() : optional_params(KernelType::ARG_MAX_MIN) {}
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
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const arg_max_min_params& params) const;
    virtual DispatchData SetDefault(const arg_max_min_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
    void SetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

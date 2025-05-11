// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cum_sum_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct cum_sum_params : public base_params {
    cum_sum_params() : base_params(KernelType::CUM_SUM), axis(CumSumAxis::BATCH), exclusive(false), reverse(false) {}

    CumSumAxis axis;
    bool exclusive;
    bool reverse;
};

class CumSumKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~CumSumKernelBase() = default;

    struct DispatchData : public CommonDispatchData {
        size_t sum_items_num;

        DispatchData() : sum_items_num(0){}
    };

protected:
    Tensor::DataChannelName GetCumSumAxis(const cum_sum_params& params) const;
    int32_t GetCumSumAxisIndex(const cum_sum_params& params) const;
    size_t GetRealAxisIndex(const cum_sum_params& params) const;
    virtual JitConstants GetJitConstants(const cum_sum_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const cum_sum_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    bool Validate(const Params&) const override;
    Datatype GetActivationType(const cum_sum_params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector

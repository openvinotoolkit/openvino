/*
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cum_sum_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct cum_sum_optional_params : optional_params {
    cum_sum_optional_params() : optional_params(KernelType::CUM_SUM) {}
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
    ParamsKey GetSupportedKey() const override;
    virtual JitConstants GetJitConstants(const cum_sum_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const cum_sum_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
    bool Validate(const Params&, const optional_params&) const override;
    Datatype GetActivationType(const cum_sum_params& params) const;
};
}  // namespace kernel_selector

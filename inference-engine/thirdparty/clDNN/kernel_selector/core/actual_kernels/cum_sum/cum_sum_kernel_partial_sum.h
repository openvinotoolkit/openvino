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

#include "cum_sum_kernel_base.h"

namespace kernel_selector {
class CumSumKernelPartialSum : public CumSumKernelBase {
public:
    CumSumKernelPartialSum() : CumSumKernelBase("cum_sum_partial_sum") {}
    virtual ~CumSumKernelPartialSum() = default;
protected:
    struct MultiDispatchData {
        DispatchData stage_1;
        DispatchData stage_final;
    };

    JitConstants GetJitConstants(const cum_sum_params& params, DispatchData dispatchData) const override;
    KernelsData GetMultiStageKernelsData(const Params& params, const optional_params&, float estimated_time) const;
    MultiDispatchData SetDefaultForMulti(const cum_sum_params& params) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector

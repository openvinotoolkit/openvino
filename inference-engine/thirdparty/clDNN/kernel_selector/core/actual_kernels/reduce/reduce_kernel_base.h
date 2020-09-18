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

#pragma once

#include "common_kernel_base.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reduce_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reduce_params : public base_params {
    reduce_params() : base_params(KernelType::REDUCE), reduceMode(ReduceMode::MAX), keepDims(0) {}

    ReduceMode reduceMode;
    std::vector<uint16_t> reduceAxes;
    int32_t keepDims;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reduce_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reduce_optional_params : optional_params {
    reduce_optional_params() : optional_params(KernelType::REDUCE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ReduceKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ReduceKernelBase : public common_kernel_base {
public:
    using common_kernel_base::common_kernel_base;
    using DispatchData = CommonDispatchData;

    virtual ~ReduceKernelBase() {}

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const reduce_params& params) const;
    virtual CommonDispatchData SetDefault(const reduce_params& params, const optional_params&) const = 0;
    Datatype GetAccumulatorType(const reduce_params& p) const;
    Datatype GetFinalAccumulatorType(const reduce_params& p) const;
    Datatype GetActivationType(const reduce_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimatedTime) const;
};
}  // namespace kernel_selector

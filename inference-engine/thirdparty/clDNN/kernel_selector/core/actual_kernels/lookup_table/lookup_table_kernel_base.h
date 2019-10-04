/*
// Copyright (c) 2018 Intel Corporation
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

#include "common_kernel_base.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lookup_table_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lookup_table_params : public base_params {
    lookup_table_params() : base_params(KernelType::LOOKUP_TABLE) {}

    LookUpTableAxis lookUpTableAxis = LookUpTableAxis::XYF;
    uint32_t numberOfValues = 0;
    DataTensor inputIndices;

    virtual ParamsKey GetParamsKey() const {
        ParamsKey k = base_params::GetParamsKey();
        k.EnableLookUpTableAxis(lookUpTableAxis);
        k.EnableLookUpTableIndicesFormat(inputIndices.GetDType());
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lookup_table_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lookup_table_optional_params : optional_params {
    lookup_table_optional_params() : optional_params(KernelType::LOOKUP_TABLE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lookup_table_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class LookUpTableKernelBase : public common_kernel_base {
public:
    using common_kernel_base::common_kernel_base;
    virtual ~LookUpTableKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const lookup_table_params& params) const;
    virtual DispatchData SetDefault(const lookup_table_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimatedTime) const;
};
}  // namespace kernel_selector
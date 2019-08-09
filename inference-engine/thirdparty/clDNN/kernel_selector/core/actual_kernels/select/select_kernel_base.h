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

#pragma once

#include "common_kernel_base.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// select_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct select_params : public base_params {
    select_params() : base_params(KernelType::SELECT) {}

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// select_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct select_optional_params : optional_params {
    select_optional_params() : optional_params(KernelType::SELECT) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SelectKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SelectKernelBase : public common_kernel_base {
public:
    using common_kernel_base::common_kernel_base;
    virtual ~SelectKernelBase() {}

    using DispatchData = CommonDispatchData;
    JitConstants GetJitConstantsCommon(const select_params& params) const;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    virtual JitConstants GetJitConstants(const select_params& params) const;
    virtual DispatchData SetDefault(const select_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params& options) const;
};
}  // namespace kernel_selector

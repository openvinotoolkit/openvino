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
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// max_unpooling_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct max_unpooling_params : public base_params {
    max_unpooling_params() : base_params(KernelType::MAX_UNPOOLING) {}

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// max_unpooling_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct max_unpooling_optional_params : optional_params {
    max_unpooling_optional_params() : optional_params(KernelType::MAX_UNPOOLING) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MaxUnpoolingKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MaxUnpoolingKernelBase : public common_kernel_base {
public:
    using common_kernel_base::common_kernel_base;
    virtual ~MaxUnpoolingKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        bool needsBoundary = false;
    };

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const max_unpooling_params& params) const;
    virtual DispatchData SetDefault(const max_unpooling_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimatedTime) const;
};
}  // namespace kernel_selector
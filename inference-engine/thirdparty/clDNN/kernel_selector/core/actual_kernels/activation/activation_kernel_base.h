// Copyright (c) 2016 Intel Corporation
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
// activation_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct activation_params : public base_params {
    activation_params() : base_params(KernelType::ACTIVATION) {}

    MultiDataTensor inputActivationParams;

    virtual ParamsKey GetParamsKey() const {
        auto k = base_params::GetParamsKey();
        if (!inputActivationParams.empty()) {
            k.EnableActivationAdditionalParamsAsInput();
        }
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// activation_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct activation_optional_params : optional_params {
    activation_optional_params() : optional_params(KernelType::ACTIVATION) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ActivationKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ActivationKernelBase : public common_kernel_base {
public:
    using DispatchData = CommonDispatchData;
    using common_kernel_base::common_kernel_base;

    virtual ~ActivationKernelBase() {}

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    virtual JitConstants GetJitConstants(const activation_params& params, DispatchData kd) const;
    virtual DispatchData SetDefault(const activation_params& arg) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params& options) const;
};
}  // namespace kernel_selector
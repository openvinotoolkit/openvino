// Copyright (c) 2019 Intel Corporation
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


namespace kernel_selector
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // contract_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct contract_params : public base_params
    {
        contract_params()
            : base_params(KernelType::CONTRACT)
        {
        }
        ContractMode mode;
        std::vector<uint16_t> reduction_axes;

    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // contract_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct contract_optional_params : optional_params
    {
        contract_optional_params()
            : optional_params(KernelType::CONTRACT)
        {
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ContractKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ContractKernelBase : public common_kernel_base
    {
    public:
        using common_kernel_base::common_kernel_base;

        using DispatchData = CommonDispatchData;

    protected:
        static JitConstants GetJitConstants(const contract_params& params);
        static DispatchData SetDefault(const contract_params& params);
        KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimated_time) const;
    };
}

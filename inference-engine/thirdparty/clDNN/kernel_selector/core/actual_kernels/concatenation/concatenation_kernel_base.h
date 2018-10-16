/*
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
*/

#pragma once

#include "common_kernel_base.h"
#include "kernel_selector_params.h"

namespace kernel_selector 
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // concatenation_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct concatenation_params : public base_params
    {
        concatenation_params() : base_params(KernelType::CONCATENATION) {}

        ConcatAxis axis = ConcatAxis::FEATURE;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = base_params::GetParamsKey();
            k.EnableConcatAxis(axis);
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // concatenation_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct concatenation_optional_params : optional_params
    {
        concatenation_optional_params() : optional_params(KernelType::CONCATENATION) {}
        bool kernelPerInput = true;

        virtual ParamsKey GetSupportedKey() const
        {
            ParamsKey k = optional_params::GetSupportedKey();

            if (kernelPerInput)
            {
                k.EnableConcatKernelPerInput();
            }
            else
            {
                k.EnableConcatOneKernel();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConcatenationKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ConcatenationKernelBase : public common_kernel_base
    {
    public:
        using common_kernel_base::common_kernel_base;
        virtual ~ConcatenationKernelBase() {}

        using DispatchData = CommonDispatchData;
    
    protected:
        virtual bool Validate(const Params&, const optional_params&) const override;
        virtual JitConstants GetJitConstants(const concatenation_params& params) const;
        virtual DispatchData SetDefault(const concatenation_params& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
    };
}
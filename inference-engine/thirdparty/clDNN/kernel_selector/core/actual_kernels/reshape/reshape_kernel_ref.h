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
 
namespace kernel_selector 
{    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // reshape_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct reshape_params : public base_params
    {
        reshape_params() : base_params(KernelType::RESHAPE) {}

        virtual ParamsKey GetParamsKey() const
        {
            return base_params::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // reshape_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct reshape_optional_params : optional_params
    {
        reshape_optional_params() : optional_params(KernelType::RESHAPE) {}
    };

    class ReshapeKernelRef : public common_kernel_base
    {
    public:
        ReshapeKernelRef() : common_kernel_base("reshape_ref") {}
        virtual ~ReshapeKernelRef() {}

        virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
        virtual ParamsKey GetSupportedKey() const override;
    };
}
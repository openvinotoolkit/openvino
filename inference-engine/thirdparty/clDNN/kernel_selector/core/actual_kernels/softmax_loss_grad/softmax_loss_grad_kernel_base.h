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

namespace kernel_selector 
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftmaxLossGradParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct softmax_loss_grad_params : public base_params
    {
        softmax_loss_grad_params() : base_params(KernelType::SOFT_MAX_LOSS_GRAD) {}

        virtual ParamsKey GetParamsKey() const
        {
            return base_params::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftmaxLossGradOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct softmax_loss_grad_optional_params : optional_params
    {
        softmax_loss_grad_optional_params() : optional_params(KernelType::SOFT_MAX_LOSS_GRAD) {}
    };

    class SoftmaxLossGradKernelBase : public common_kernel_base
    {
    public:
        using common_kernel_base::common_kernel_base;
        virtual ~SoftmaxLossGradKernelBase() {}

    protected:
        virtual bool Validate(const Params&, const optional_params&) const;
        virtual JitConstants GetJitConstants(const softmax_loss_grad_params& params) const;
        virtual CommonDispatchData SetDefault(const softmax_loss_grad_params& params, const optional_params& optParams) const;
        KernelsData GetCommonKernelsData(const Params& params, const optional_params& optParams) const;
    };
}
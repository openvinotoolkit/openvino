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

#include "training_kernel_base.h"
#include "kernel_selector_params.h"

namespace kernel_selector
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // scale_grad_weights_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct scale_grad_weights_params : public training_params
    {
        scale_grad_weights_params() : training_params(KernelType::SCALE_GRAD_WEIGHTS) {}

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = training_params::GetParamsKey();

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // scale_grad_weights_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct scale_grad_weights_optional_params : training_optional_params
    {
        scale_grad_weights_optional_params() : training_optional_params(KernelType::SCALE_GRAD_WEIGHTS) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ScaleGradWeightsKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ScaleGradWeightsKernelBase : public training_kernel_base
    {
    public:
        using training_kernel_base::training_kernel_base;
        virtual ~ScaleGradWeightsKernelBase() {}

        using DispatchData = CommonDispatchData;

    protected:
        virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const;
        virtual JitConstants GetJitConstants(const scale_grad_weights_params& params) const;
        virtual DispatchData SetDefault(const scale_grad_weights_params& params) const;
    };
}
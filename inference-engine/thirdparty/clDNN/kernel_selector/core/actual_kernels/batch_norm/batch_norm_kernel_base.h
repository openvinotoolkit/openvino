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
    // batch_norm_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct batch_norm_params : public base_params
    {
        batch_norm_params() : base_params(KernelType::BATCH_NORM_GRAD) {}

        struct DedicatedParams
        {
            float epsilon;
            bool with_inv_var;
			bool with_scale_shift;
			bool with_mean_var_out = false;
        };

        DedicatedParams batchNormParams;

        virtual ParamsKey GetParamsKey() const
        {
            return base_params::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // batch_norm_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct batch_norm_optional_params : optional_params
    {
        batch_norm_optional_params() : optional_params(KernelType::BATCH_NORM_GRAD) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // BatchNormKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class BatchNormKernelBase : public common_kernel_base
    {
    public:
        using common_kernel_base::common_kernel_base;
        virtual ~BatchNormKernelBase() {}

        using DispatchData = CommonDispatchData;

    protected:
        virtual bool Validate(const Params& params, const optional_params& options) const override;
        KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimatedTime) const;
        virtual JitConstants GetJitConstants(const batch_norm_params& params) const;
        virtual DispatchData SetDefault(const batch_norm_params& params) const;
    };
}
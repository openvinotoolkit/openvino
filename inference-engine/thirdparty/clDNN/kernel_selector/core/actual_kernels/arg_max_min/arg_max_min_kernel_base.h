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
    // arg_max_min_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct arg_max_min_params : public base_params
    {
        arg_max_min_params() : base_params(KernelType::ARG_MAX_MIN) {}

        ArgMaxMinAxis	argMaxMinAxis = ArgMaxMinAxis::XYF;
        ArgMaxMinOut	argMaxMinOut = ArgMaxMinOut::MAX;
        uint32_t		topK = 1;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = base_params::GetParamsKey();
            k.EnableArgMaxMinAxis(argMaxMinAxis);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // arg_max_min_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct arg_max_min_optional_params : optional_params
    {
        arg_max_min_optional_params() : optional_params(KernelType::ARG_MAX_MIN) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ArgMaxMinKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class ArgMaxMinKernelBase : public common_kernel_base
	{
	public:
		using common_kernel_base::common_kernel_base;
		virtual ~ArgMaxMinKernelBase() {}

		struct DispatchData : public CommonDispatchData
		{
		};

	protected:
		virtual bool Validate(const Params&, const optional_params&) const override;
		virtual JitConstants GetJitConstants(const arg_max_min_params& params) const;
		virtual DispatchData SetDefault(const arg_max_min_params& params) const;
		KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimatedTime) const;
	};
}
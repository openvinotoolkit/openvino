/*
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
*/

#pragma once

#include "common_kernel_base.h"

namespace kernel_selector
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // depth_to_space_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct depth_to_space_params : public base_params
    {
        depth_to_space_params() : base_params(KernelType::DEPTH_TO_SPACE) {}

        size_t block_size;

        virtual ParamsKey GetParamsKey() const
        {
            return base_params::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // depth_to_space_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct depth_to_space_optional_params : optional_params
    {
        depth_to_space_optional_params() : optional_params(KernelType::DEPTH_TO_SPACE) {}
    };

    class DepthToSpaceKernelRef : public common_kernel_base
    {
    public:
        DepthToSpaceKernelRef() : common_kernel_base("depth_to_space_ref") {}
        virtual ~DepthToSpaceKernelRef() {}
        virtual JitConstants GetJitConstants(const depth_to_space_params& params) const;
        virtual CommonDispatchData SetDefault(const depth_to_space_params& params, const optional_params&) const;
        virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
        virtual ParamsKey GetSupportedKey() const override;
    };
}

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


namespace kernel_selector 
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // index_select_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct index_select_params : public base_params
    {
        index_select_params()
            : base_params(KernelType::INDEX_SELECT)
        {}

        std::vector<IndexSelectAxis> axes = { IndexSelectAxis::BATCH };
        bool reverse = false;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // index_select_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct index_select_optional_params : optional_params
    {
        index_select_optional_params()
            : optional_params(KernelType::INDEX_SELECT)
        {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // IndexSelectKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class IndexSelectKernelBase : public common_kernel_base
    {
    public:
        using common_kernel_base::common_kernel_base;
        virtual ~IndexSelectKernelBase() {}

        using DispatchData = CommonDispatchData;

    protected:
        static JitConstants GetJitConstants(const index_select_params& params);
        static DispatchData SetDefault(const index_select_params& params);
        KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimated_time) const;
    };
}

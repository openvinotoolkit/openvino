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
    // roi_pooling_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct roi_pooling_params : public base_params
    {
        roi_pooling_params() : base_params(KernelType::ROI_POOLING) {}

        PoolType    mode = PoolType::MAX;
        size_t      pooledWidth = 0;
        size_t      pooledHeight = 0;
        size_t      groupSize = 0;
        float       spatialScale = 1.f;

        virtual ParamsKey GetParamsKey() const
        {
            return base_params::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // roi_pooling_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct roi_pooling_optional_params : optional_params
    {
        roi_pooling_optional_params() : optional_params(KernelType::ROI_POOLING) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ROIPoolingKernelRef
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ROIPoolingKernelRef : public common_kernel_base
    {
    public:
        ROIPoolingKernelRef() : common_kernel_base("roi_pooling_ref") {}
        virtual ~ROIPoolingKernelRef() {}

        using DispatchData = CommonDispatchData;

        virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
        virtual ParamsKey GetSupportedKey() const override;

    protected:
        JitConstants GetJitConstants(const roi_pooling_params& params) const;
    };
}
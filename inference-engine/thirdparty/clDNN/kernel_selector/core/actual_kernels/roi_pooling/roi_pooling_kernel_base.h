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

#include <iostream>
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
        bool        position_sensitive = false;
        int         pooledWidth = 0;
        int         pooledHeight = 0;
        int         spatial_bins_x = 1;
        int         spatial_bins_y = 1;
        float       spatialScale = 1.f;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = base_params::GetParamsKey();
            if (position_sensitive)
            {
                k.EnablePositionSensitivePooling();
            }
            k.EnablePoolType(mode);

            return k;
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
    // ROIPoolingKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ROIPoolingKernelBase : public common_kernel_base
    {
    public:
        using common_kernel_base::common_kernel_base;
        virtual ~ROIPoolingKernelBase() {};

        using DispatchData = CommonDispatchData;

        KernelsData GetCommonKernelsData(const Params& params, const optional_params& options, float estimatedTime) const;
    protected:
        virtual JitConstants GetJitConstants(const roi_pooling_params& params) const;
    };
}

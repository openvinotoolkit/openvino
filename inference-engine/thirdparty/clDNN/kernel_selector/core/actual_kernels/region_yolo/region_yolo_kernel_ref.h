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
    // region_yolo_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct region_yolo_params : public base_params
    {
        region_yolo_params() : base_params(KernelType::REGION_YOLO) {}

        uint32_t coords;
        uint32_t classes;
        uint32_t num;
        uint32_t mask_size;
        bool do_softmax;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = base_params::GetParamsKey();
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // region_yolo_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct region_yolo_optional_params : optional_params
    {
        region_yolo_optional_params() : optional_params(KernelType::REGION_YOLO) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // RegionYoloKernelRef
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class RegionYoloKernelRef : public common_kernel_base
    {
    public:
        RegionYoloKernelRef() : common_kernel_base("region_yolo_gpu_ref") {}
        virtual ~RegionYoloKernelRef() {}

        using DispatchData = CommonDispatchData;        
        virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;


    protected:
        virtual ParamsKey GetSupportedKey() const override;
        virtual JitConstants GetJitConstants(const region_yolo_params& params) const;
    };
}

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

#include "weight_bias_kernel_base.h"
#include "fully_connected_params.h"

namespace kernel_selector 
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FullyConnectedKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class FullyConnectedKernelBase : public WeightBiasKernelBase
    {
    public:
        using WeightBiasKernelBase::WeightBiasKernelBase;
        virtual ~FullyConnectedKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {
            uint32_t    unit_byte_size;
            const char* chunk_type;
            uint32_t    chunk_byte_size;
            uint32_t    units_per_chunk;
            uint32_t    bytes_per_sg_read;
            uint32_t    units_per_sg_read;
            uint32_t    responses_per_sg_exec;
            uint32_t    in_chunk_prefetch_size;
            uint32_t    filter_chunk_prefetch_size;

            uint32_t    last_rg_size;
            uint32_t    rg_count;
        };
    
        std::string GetAutoTuneOptions(int autoTuneIndex) const;
        std::vector<std::string> autoTuneOptions = { DEFAULT, NO_PRERA_SCH, AGE_BASED };
        virtual KernelsData GetTunedKernelsDataByIndex(const Params& params, const optional_params& options, DataLayout dl, std::vector<WeightsLayout> wl, float estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE, int autoTuneIndex = -1) const ;

    protected:
        virtual JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& kd) const;
        virtual DispatchData SetDefault(const fully_connected_params& params, int autoTuneIndex = -1) const;
        KernelsData GetCommonKernelsData(const Params& params, const optional_params& optParams, DataLayout dl, std::vector<WeightsLayout> wl, float estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE, const std::string exeMode = DEFAULT, int autoTuneIndex = -1) const;

        bool Validate(const Params& p, const optional_params&) const override
        {
            if (p.GetType() != KernelType::FULLY_CONNECTED)
            {
                return false;
            }

            return true;
        }
    };
}

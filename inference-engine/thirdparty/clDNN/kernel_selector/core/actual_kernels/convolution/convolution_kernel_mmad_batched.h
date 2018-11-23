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

#include "convolution_kernel_base.h"
 
namespace kernel_selector {
    
    class ConvolutionKernel_mmad_batched : public ConvolutionKernelBase
    {
    public:
        using Parent = ConvolutionKernelBase;
        ConvolutionKernel_mmad_batched() : ConvolutionKernelBase("convolution_gpu_mmad_batched") {}
        virtual ~ConvolutionKernel_mmad_batched() {}

        virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
        virtual ParamsKey GetSupportedKey() const override;

    protected:
        JitConstants GetJitConstants(const convolution_params& params, const DispatchData& kd) const override;
        DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
        virtual std::vector<WeightsLayout> GetSupportedWeightLayouts(const convolution_params&) const override
        {
            return{
                WeightsLayout::os_is_yx_isa8_osv8_isv4,
            };
        }
    };
}
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

#include "convolution_kernel_base.h"

// Step 0: 
//
// 1. choose a tutorial mode
// 2. modify convolution_tutorial.cl as well

#define ADVANCED_TUTORIAL       // simple runnable example with explanations
#ifndef ADVANCED_TUTORIAL
#define BASIC_TUTORIAL          // Skeleton to add a new kernel
#endif
 
namespace kernel_selector {
    
    class ConvolutionKernel_Tutorial : public ConvolutionKernelBase
    {
    public:
        using Parent = ConvolutionKernelBase;
        ConvolutionKernel_Tutorial() : Parent("convolution_tutorial") {}
        virtual ~ConvolutionKernel_Tutorial() {}

        virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;

    protected:
        virtual ParamsKey GetSupportedKey() const override;
        virtual std::vector<WeightsLayout> GetSupportedWeightLayouts(const convolution_params&) const override
        {
            return{
                WeightsLayout::oiyx,
                WeightsLayout::yxio,
                WeightsLayout::iyxo,
                WeightsLayout::oyxi,
            };
        }

#ifdef ADVANCED_TUTORIAL
        bool         Validate(const Params& p, const optional_params& o)                 const override;
        JitConstants GetJitConstants(const convolution_params& params, const DispatchData& kd)  const override;
        DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1)   const override;
#endif
    };
}

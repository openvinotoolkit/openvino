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

#include "activation_kernel_base.h"
 
// Step 0: 
//
// 1. choose a tutorial mode
// 2. modify activation_tutorial.cl as well

#define ADVANCED_TUTORIAL       // simple runnable example with explanations
#ifndef ADVANCED_TUTORIAL
#define BASIC_TUTORIAL          // Skeleton to add a new kernel
#endif

namespace kernel_selector {
    
    class ActivationKernel_Tutorial : public ActivationKernelBase
    {
    public:
        using Parent = ActivationKernelBase;
        ActivationKernel_Tutorial() : Parent("activation_tutorial") {}
        virtual ~ActivationKernel_Tutorial() {}

        virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;

    protected:
        virtual ParamsKey GetSupportedKey() const override;
#ifdef ADVANCED_TUTORIAL
        virtual DispatchData SetDefault(const activation_params& arg) const override;
        virtual bool Validate(const Params& p, const optional_params& o) const override;
        virtual JitConstants GetJitConstants(const activation_params& params, DispatchData) const override;
#endif
    };
}

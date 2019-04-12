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

#include "weight_bias_kernel_base.h"
#include "embed_params.h"
#include "common_kernel_base.h"

namespace kernel_selector
{

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EmbedKernelRef
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class EmbedKernelRef : public WeightBiasKernelBase
    {
    public:
        EmbedKernelRef() : WeightBiasKernelBase("embed_ref") {}
        virtual ~EmbedKernelRef() {}

        struct DispatchData : public CommonDispatchData
        {
        };

    protected:
        virtual ParamsKey GetSupportedKey() const override;
        virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
        virtual JitConstants GetJitConstants(const embed_params& params) const;
        virtual DispatchData SetDefault(const embed_params& params) const;
    };
}

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

#include "fused_conv_bn_scale_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
    
    ParamsKey fused_conv_bn_scale_kernel_ref::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableSplitSupport();
        k.EnableBatching();
        k.DisableTuning();
        return k;
    }

    fused_conv_bn_scale_kernel_base::DispatchData fused_conv_bn_scale_kernel_ref::SetDefault(const fused_conv_bn_scale_params& arg) const
    {
        DispatchData runInfo = fused_conv_bn_scale_kernel_base::SetDefault(arg);

        runInfo.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        runInfo.gws0 = arg.output.Batch().v;
        runInfo.gws1 = arg.output.Feature().v; 
        runInfo.gws2 = 1;

        runInfo.lws0 = std::min(std::max(runInfo.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (runInfo.gws0 % runInfo.lws0 != 0)
        {
            --runInfo.lws0;
        }
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        return runInfo;
    }

    JitConstants fused_conv_bn_scale_kernel_ref::GetJitConstants(const fused_conv_bn_scale_params& params, const DispatchData& runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);

        return jit;
    }

    KernelsData fused_conv_bn_scale_kernel_ref::GetKernelsData(const Params& params, const optional_params& options) const
    {
        KernelsData kd = GetCommonKernelsData(params, options, DONT_USE_IF_HAVE_SOMETHING_ELSE);
 
        return kd;
    }
}
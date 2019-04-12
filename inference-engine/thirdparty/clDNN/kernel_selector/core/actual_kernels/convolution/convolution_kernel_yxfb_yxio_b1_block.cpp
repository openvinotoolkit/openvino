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

#include "convolution_kernel_yxfb_yxio_b1_block.h"

namespace kernel_selector 
{

    ParamsKey ConvolutionKernel_yxfb_yxio_b1_block::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::yxfb);
        k.EnableOutputLayout(DataLayout::yxfb);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableDilation();
        k.EnableSubGroup();
        return k;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b1_block::SetDefault(const convolution_params& arg, int) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);
        // TODO: fill the proper data here (I don't know where can I locate it).
        return runInfo;
    }

    JitConstants ConvolutionKernel_yxfb_yxio_b1_block::GetJitConstants(const convolution_params& params, const DispatchData& kd) const
    {
        auto cldnn_jit = ConvolutionKernelBase::GetJitConstants(params, kd);

        cldnn_jit.AddConstant(MakeJitConstant("LOCAL_WORK_GROUP_SIZE", kd.lws0));
        return cldnn_jit;
    }

    KernelsData ConvolutionKernel_yxfb_yxio_b1_block::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetTunedKernelsDataByIndex(params, options);
    }
}
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

#include "convolution_kernel_byxf_af32_depthwise.h"

namespace kernel_selector {

ParamsKey ConvolutionKernel_byxf_af32_depthiwise::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::byxf_af32);
    k.EnableOutputLayout(DataLayout::byxf_af32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableDepthwiseSeparableOpt();
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.DisableTuning();
    return k;
}

bool ConvolutionKernel_byxf_af32_depthiwise::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_byxf_af32_depthiwise::GetJitConstants(const convolution_params& params,
                                                            const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = {"", {"b", "of", "y", "x"}, "res", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return jit;
}


KernelsData ConvolutionKernel_byxf_af32_depthiwise::GetKernelsData(const Params& params,
                                                                   const optional_params& options) const {
    KernelsData kd = GetTunedKernelsDataByIndex(params, options);
    if (!kd.empty())
        kd[0].estimatedTime = FORCE_PRIORITY_3;
    return kd;
}

}  // namespace kernel_selector

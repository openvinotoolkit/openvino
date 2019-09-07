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

#include "deformable_convolution_kernel_bfyx_interp.h"
#include <string>

namespace kernel_selector {

ParamsKey DeformableConvolutionKernel_bfyx_interp::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.DisableTuning();
    k.EnableLocalConvolution();
    k.EnableGroupedConvolution();
    k.EnableDeformableMode();
    return k;
}

CommonDispatchData DeformableConvolutionKernel_bfyx_interp::SetDefault(const convolution_params& params) const {
    CommonDispatchData kd;

    const auto& out = params.output;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto b = out.Batch().v;
    auto kernel_size = params.kernelSize.x * params.kernelSize.y;

    kd.gws0 = Align(x * y, 16);
    kd.gws1 = params.deformable_groups * b;
    kd.gws2 = kernel_size;

    kd.lws0 = 16;
    kd.lws1 = 1;
    kd.lws2 = 1;

    kd.effiency = FORCE_PRIORITY_2;

    return kd;
}


JitConstants DeformableConvolutionKernel_bfyx_interp::GetJitConstants(const convolution_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", 16));
    jit.AddConstant(MakeJitConstant("FILTER_SIZE_X", params.kernelSize.x));
    jit.AddConstant(MakeJitConstant("FILTER_SIZE_Y", params.kernelSize.y));
    jit.AddConstants({MakeJitConstant("STRIDE", params.stride),
                      MakeJitConstant("PADDING", params.padding),
                      MakeJitConstant("DILATION", params.dilation)
                     });
    jit.AddConstants({MakeJitConstant("DEFORMABLE_GROUPS", params.deformable_groups)});
    jit.AddConstants({MakeJitConstant("DEFORMABLE_MODE", params.deformable_mode)});
    return jit;
}

KernelsData DeformableConvolutionKernel_bfyx_interp::GetKernelsData(const Params& params,
                                                                    const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    CommonDispatchData runInfo = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point, DEFAULT,
                     false, false, (uint32_t)newParams.inputs.size());

    return {kd};
}
}  // namespace kernel_selector

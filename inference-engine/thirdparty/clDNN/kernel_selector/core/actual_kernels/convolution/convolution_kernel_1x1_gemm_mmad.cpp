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

#include "convolution_kernel_1x1_gemm_mmad.h"

namespace kernel_selector {

ParamsKey ConvolutionKernel_1x1_gemm_MMAD::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::byxf_af32);
    k.EnableOutputLayout(DataLayout::byxf_af32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.DisableTuning();
    return k;
}

bool ConvolutionKernel_1x1_gemm_MMAD::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    if (params.filterSize.x != 1 || params.filterSize.y != 1)
        return false;

    if (params.stride.x != 1 || params.stride.y != 1)
        return false;

    if (params.padding.x != 0 || params.padding.y != 0)
        return false;

    const auto& input = params.inputs[0];

    // we do not support padded input
    if (input.X().pad.Total() != 0 || input.Y().pad.Total() != 0)
        return false;

    if (params.split != 1)
        return false;

    return true;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_1x1_gemm_MMAD::SetDefault(const convolution_params& arg, int) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

    // Sub-group size used by "convolution_1x1_gemm_MMAD" kernel.
    constexpr size_t sub_group_size = 8;

    const auto of_maps = arg.output.Feature().v;
    const size_t of_threads_per_batch = RoundUp(of_maps, sub_group_size);

    runInfo.effiency = FORCE_PRIORITY_2;

    runInfo.gws0 = RoundUp(arg.output.X().v * arg.output.Y().v, 8) / 8;
    runInfo.gws1 = of_threads_per_batch * arg.output.Batch().v;
    runInfo.gws2 = 1;

    runInfo.lws0 = 1;
    runInfo.lws1 = sub_group_size;
    runInfo.lws2 = 1;

    return runInfo;
}

JitConstants ConvolutionKernel_1x1_gemm_MMAD::GetJitConstants(const convolution_params& params, const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws1));

    // pitch for special block format used in this kernel
    const size_t ifm_32_aligned = Align(params.weights.IFM().v, 32);
    const size_t filter_ofm_block_pitch = (ifm_32_aligned / 32) * params.weights.X().v * params.weights.Y().v * 4 * 8 * 8;
    jit.AddConstant(MakeJitConstant("FILTER_OFM_BLOCK_PITCH", filter_ofm_block_pitch));

    return jit;
}

KernelsData ConvolutionKernel_1x1_gemm_MMAD::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector

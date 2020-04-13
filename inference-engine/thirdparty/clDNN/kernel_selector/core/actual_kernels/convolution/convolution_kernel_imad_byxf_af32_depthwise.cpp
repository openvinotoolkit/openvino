/*
// Copyright (c) 2016-2020 Intel Corporation
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

#include "convolution_kernel_imad_byxf_af32_depthwise.h"

#define SIMD_SIZE 16

namespace kernel_selector {

ParamsKey ConvolutionKernel_imad_byxf_af32_depthiwise::GetSupportedKey() const {
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
    k.EnableGroupedConvolution();
    return k;
}

static size_t GetTileLength(size_t out_x) {
    for (int i = 20; i >= 1; i--) {
        if (out_x % i == 0)
            return i;
    }
    return 1;
}

static int GetSplit(size_t out_x, int stride) {
    if (out_x >= 75) {
        if (stride > 1)
            return 1;
        else
            return 3;
    }

    if (out_x == 38 && stride == 2)
        return 2;

    if (out_x < 75) {
        if (stride > 1)
            return 1;
        else if (out_x % 2 == 0)
            return 2;
    }
    return 1;
}

bool ConvolutionKernel_imad_byxf_af32_depthiwise::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    const convolution_params& cp = static_cast<const convolution_params&>(p);
    if (cp.inputs[0].Feature().v != cp.groups || cp.output.Feature().v != cp.groups || cp.groups == 1) {
        return false;
    }

    return true;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_imad_byxf_af32_depthiwise::SetDefault(const convolution_params& arg,
    int) const {
    DispatchData runInfo = Parent::SetDefault(arg);

    runInfo.efficiency = FORCE_PRIORITY_1;

    runInfo.gws0 = Align(arg.output.Feature().v, SIMD_SIZE) * arg.output.Batch().v;
    runInfo.gws1 = arg.output.X().v / GetTileLength(arg.output.X().v);
    runInfo.gws2 = CeilDiv(arg.output.Y().v, GetSplit(arg.output.Y().v, arg.stride.y));

    std::vector<size_t> local = { SIMD_SIZE, 1, 1 };

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    return runInfo;
}

JitConstants ConvolutionKernel_imad_byxf_af32_depthiwise::GetJitConstants(const convolution_params& params,
                                                            const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("ALIGNED_OFM", Align(params.output.Feature().v, SIMD_SIZE)));
    jit.AddConstant(MakeJitConstant("OUT_BLOCK_WIDTH", GetTileLength(params.output.X().v)));
    jit.AddConstant(MakeJitConstant("SPLIT_Y", GetSplit(params.output.Y().v, params.stride.y)));
    jit.AddConstant(MakeJitConstant("SIMD_SIZE", SIMD_SIZE));

    if (params.output.Y().v % GetSplit(params.output.Y().v, params.stride.y) != 0)
        jit.AddConstant(MakeJitConstant("SPLIT_LEFTOVERS", params.output.Y().v % GetSplit(params.output.Y().v, params.stride.y)));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = {"", {"b", "of", "(y+m)", "(x+l)"}, "res", input_dt, 1 };
        conf_scalar.SetLoopAxes({Tensor::DataChannelName::Y, Tensor::DataChannelName::X});
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return jit;
}


KernelsData ConvolutionKernel_imad_byxf_af32_depthiwise::GetKernelsData(const Params& params,
                                                                   const optional_params& options) const {
    KernelsData kd = GetTunedKernelsDataByIndex(params, options);
    if (!kd.empty())
        kd[0].estimatedTime = FORCE_PRIORITY_1;
    return kd;
}

}  // namespace kernel_selector

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

#include "convolution_kernel_winograd_6x3_s1_fused.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

ParamsKey ConvolutionKernel_Winograd_6x3_s1_fused::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();

    return k;
}

JitConstants ConvolutionKernel_Winograd_6x3_s1_fused::GetJitConstants(const convolution_params& params,
                                                                      const DispatchData& runInfo) const {
    JitConstants jit = Parent::GetJitConstants(params, runInfo);

    const auto idepth = params.inputs[0].Feature().v;
    const auto input_pad_y = params.inputs[0].Y().pad.before + params.inputs[0].Y().pad.after;
    const auto input_pad_x = params.inputs[0].X().pad.before + params.inputs[0].X().pad.after;
    const auto rows = params.inputs[0].Y().v + input_pad_y;
    const auto cols = params.inputs[0].X().v + input_pad_x;

    auto output_pad_x_before = params.output.GetDims()[0].pad.before;
    auto output_pad_y_before = params.output.GetDims()[1].pad.before;
    auto output_pad_x_after = params.output.GetDims()[0].pad.after;
    auto output_pad_y_after = params.output.GetDims()[1].pad.after;
    auto C4_up16 = ((uint32_t)((idepth + 15) / 16) * 16) / 4;

    // if there's input padding then input offset should be ignored
    const auto inoffset_x = (input_pad_x) ? 0 : params.padding.x;
    const auto inoffset_y = (input_pad_y) ? 0 : params.padding.y;

    jit.AddConstants({
        MakeJitConstant("H", rows),
        MakeJitConstant("W", cols),
        MakeJitConstant("P", rows - 3 + 1 + output_pad_y_before + output_pad_y_after + 2 * inoffset_y),
        MakeJitConstant("Q", cols - 3 + 1 + output_pad_x_before + output_pad_x_after + 2 * inoffset_x),
        MakeJitConstant("R", 3),
        MakeJitConstant("S", 3),
        MakeJitConstant("N", 1),
        MakeJitConstant("px", inoffset_x),
        MakeJitConstant("py", inoffset_y),
        MakeJitConstant("sx", 1),
        MakeJitConstant("sy", 1),
        MakeJitConstant("C_", idepth),

        MakeJitConstant("C4_up16", C4_up16),
        MakeJitConstant("TROWS", rows),
        MakeJitConstant("TCOLS", 8),
        MakeJitConstant("KROWSW", 3),
        MakeJitConstant("KCOLSW", 8),
    });

    return jit;
}

std::vector<WeightsLayout> ConvolutionKernel_Winograd_6x3_s1_fused::GetSupportedWeightLayouts(
    const convolution_params& params) const {
    // check if image weights layout will fit into device memory, if not then try to fallback to buffer
    if (CheckImageSize(params, WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb)) {
        return {WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb};
    } else {
        return {WeightsLayout::winograd_6x3_s1_fused_weights};
    }
}

ConvolutionKernel_Winograd_6x3_s1_fused::Parent::DispatchData ConvolutionKernel_Winograd_6x3_s1_fused::SetDefault(
    const convolution_params& arg,
    int) const {
    Parent::DispatchData runInfo = Parent::SetDefault(arg);

    const auto odepth = arg.output.Feature().v;
    const auto input_pad_y = arg.inputs[0].Y().pad.before + arg.inputs[0].Y().pad.after;
    const auto input_pad_x = arg.inputs[0].X().pad.before + arg.inputs[0].X().pad.after;
    const auto rows = arg.inputs[0].Y().v + input_pad_y;
    const auto cols = arg.inputs[0].X().v + input_pad_x;

    // if there's input padding then input offset should be ignored
    const auto inoffset_x = (input_pad_x) ? 0 : arg.padding.x;
    const auto inoffset_y = (input_pad_y) ? 0 : arg.padding.y;

    auto P = rows - 2 + 2 * inoffset_y;
    auto Q = cols - 2 + 2 * inoffset_x;
    auto K = odepth;
    auto N = 1;

    uint32_t global_step[3] = {14, 6, 16 * 8};
    uint32_t local_size[3] = {16, 1, 8};

    runInfo.gws0 = ((uint32_t)((Q + global_step[0] - 1)) / global_step[0]) * local_size[0];
    runInfo.gws1 = ((uint32_t)((P + global_step[1] - 1)) / global_step[1]) * local_size[1];
    runInfo.gws2 = ((uint32_t)((N * K * 8 + global_step[2] - 1)) / global_step[2]) * local_size[2];

    runInfo.lws0 = local_size[0];
    runInfo.lws1 = local_size[1];
    runInfo.lws2 = local_size[2];

    runInfo.effiency = FORCE_PRIORITY_1;

    return runInfo;
}

bool ConvolutionKernel_Winograd_6x3_s1_fused::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    const convolution_params& params = static_cast<const convolution_params&>(p);

    if ((params.weights.X().v != 3) || (params.weights.Y().v != 3) || (params.stride.x != 1) ||
        (params.stride.y != 1) || (params.filterSize.x != 3) || (params.filterSize.y != 3) ||
        (params.output.Feature().v % 32) || (params.inputs[0].Feature().v % 32) ||
        (params.output.Feature().pad.before != 0) || (params.output.Feature().pad.after != 0) ||
        (params.output.Batch().pad.before != 0) || (params.output.Batch().pad.after != 0) ||
        // TODO: add support to batch > 1
        (params.inputs[0].Batch().v != 1)) {
        return {};
    }

    return true;
}

KernelsData ConvolutionKernel_Winograd_6x3_s1_fused::GetKernelsData(const Params& params,
                                                                    const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector
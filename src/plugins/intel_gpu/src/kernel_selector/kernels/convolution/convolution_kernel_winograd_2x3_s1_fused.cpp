// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_winograd_2x3_s1_fused.h"

namespace kernel_selector {

ParamsKey ConvolutionKernel_Winograd_2x3_s1_fused::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
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

DeviceFeaturesKey ConvolutionKernel_Winograd_2x3_s1_fused::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_shuffle();

    return k;
}

JitConstants ConvolutionKernel_Winograd_2x3_s1_fused::GetJitConstants(const convolution_params& params,
                                                                      const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    const auto idepth = params.inputs[0].Feature().v;
    const auto input_pad_y = params.inputs[0].Y().pad.before + params.inputs[0].Y().pad.after;
    const auto input_pad_x = params.inputs[0].X().pad.before + params.inputs[0].X().pad.after;
    const auto rows = params.inputs[0].Y().v + input_pad_y;
    const auto cols = params.inputs[0].X().v + input_pad_x;

    auto output_pad_x_before = params.outputs[0].GetDims()[0].pad.before;
    auto output_pad_y_before = params.outputs[0].GetDims()[1].pad.before;
    auto output_pad_x_after = params.outputs[0].GetDims()[0].pad.after;
    auto output_pad_y_after = params.outputs[0].GetDims()[1].pad.after;
    auto C4_up16 = ((uint32_t)((idepth + 15) / 16) * 16) / 4;

    // if there's input padding then input offset should be ignored
    const auto inoffset_x = (input_pad_x) ? 0 : params.padding_begin.x;
    const auto inoffset_y = (input_pad_y) ? 0 : params.padding_begin.y;

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

        MakeJitConstant("C4_up16", C4_up16),
        MakeJitConstant("TROWS", rows),
        MakeJitConstant("TCOLS", 4),
        MakeJitConstant("KROWSW", 3),
        MakeJitConstant("KCOLSW", 4),
    });

    return jit;
}

ConvolutionKernel_Winograd_2x3_s1_fused::Parent::DispatchData ConvolutionKernel_Winograd_2x3_s1_fused::SetDefault(
    const convolution_params& arg,
    int) const {
    Parent::DispatchData dispatchData = Parent::SetDefault(arg);

    const auto odepth = arg.outputs[0].Feature().v;
    const auto input_pad_y = arg.inputs[0].Y().pad.before + arg.inputs[0].Y().pad.after;
    const auto input_pad_x = arg.inputs[0].X().pad.before + arg.inputs[0].X().pad.after;
    const auto rows = arg.inputs[0].Y().v + input_pad_y;
    const auto cols = arg.inputs[0].X().v + input_pad_x;

    // if there's input padding then input offset should be ignored
    const auto inoffset_x = (input_pad_x) ? 0 : arg.padding_begin.x;
    const auto inoffset_y = (input_pad_y) ? 0 : arg.padding_begin.y;

    auto P = rows - 2 + 2 * inoffset_y;
    auto Q = cols - 2 + 2 * inoffset_x;
    auto K = odepth;
    auto N = 1;

    size_t global_step[3] = {14, 4, 16 * 8};
    size_t local_size[3] = {8, 2, 8};

    size_t zStep = local_size[2];
    dispatchData.gws[0] = ((size_t)((Q + global_step[0] - 1)) / global_step[0]) * local_size[0];
    dispatchData.gws[1] = ((size_t)((P + global_step[1] - 1)) / global_step[1]) * local_size[1];
    dispatchData.gws[2] = ((size_t)((N * K * 8 + global_step[2] - 1)) / global_step[2]) * zStep;

    dispatchData.lws[0] = local_size[0];
    dispatchData.lws[1] = local_size[1];
    dispatchData.lws[2] = local_size[2];

    return dispatchData;
}

KernelsPriority ConvolutionKernel_Winograd_2x3_s1_fused::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

bool ConvolutionKernel_Winograd_2x3_s1_fused::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        return false;
    }

    const convolution_params& params = static_cast<const convolution_params&>(p);

    if ((params.weights.X().v != 3) || (params.weights.Y().v != 3) || (params.stride.x != 1) ||
        (params.stride.y != 1) || (params.filterSize.x != 3) || (params.filterSize.y != 3) ||
        (params.outputs[0].Feature().v % 32) || (params.inputs[0].Feature().v % 32) ||
        (params.outputs[0].Feature().pad.before != 0) || (params.outputs[0].Feature().pad.after != 0) ||
        (params.outputs[0].Batch().pad.before != 0) || (params.outputs[0].Batch().pad.after != 0) ||
        // TODO: add support to batch > 1
        (params.inputs[0].Batch().v != 1)) {
        return {};
    }

    return true;
}

KernelsData ConvolutionKernel_Winograd_2x3_s1_fused::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}
}  // namespace kernel_selector

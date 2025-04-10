// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_winograd_2x3_s1.h"

namespace kernel_selector {

constexpr uint32_t winograd_tile_n = 4;
constexpr uint32_t winograd_tile_m = 8;
constexpr uint32_t winograd_input_tile_width = 4;
constexpr uint32_t winograd_input_tile_height = 1;

ParamsKey ConvolutionKernel_Winograd_2x3_s1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::winograd_2x3_s1_data);
    k.EnableOutputLayout(DataLayout::winograd_2x3_s1_data);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_Winograd_2x3_s1::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_shuffle();

    return k;
}

JitConstants ConvolutionKernel_Winograd_2x3_s1::GetJitConstants(const convolution_params& params,
                                                                const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    const size_t input_tile_width = winograd_input_tile_width;
    const size_t input_tile_height = winograd_input_tile_height;
    const size_t winograd_filter_height =
        params.filterSize.y;  // for this format, winograd filter is considered to be a set of 1d filters so its height
                              // should remain the same as original filter's

    const size_t nr_tiles_x =
        Align(params.outputs[0].X().v, 4) / input_tile_width;  // input is already in winograd domain, so simply divide its
                                                           // width by tile's width to get tiles count
    const size_t nr_tiles_y = Align(params.outputs[0].Y().v, 8) / input_tile_height;
    const size_t total_tiles_count = nr_tiles_x * nr_tiles_y;

    jit.AddConstants({
        MakeJitConstant("INPUT0_SIZE_WINOGRAD_X", Align(params.inputs[0].X().v, 4)),
        MakeJitConstant("INPUT0_SIZE_WINOGRAD_Y", Align(params.inputs[0].Y().v - 2, 8) + 2),
        MakeJitConstant("N", params.outputs[0].Feature().v),
        MakeJitConstant("M", total_tiles_count),
        MakeJitConstant("K", params.inputs[0].Feature().v * winograd_filter_height),
    });

    return jit;
}

ConvolutionKernel_Winograd_2x3_s1::Parent::DispatchData ConvolutionKernel_Winograd_2x3_s1::SetDefault(const convolution_params& arg,
                                                                                                      int) const {
    Parent::DispatchData dispatchData = Parent::SetDefault(arg);

    const size_t tile_n = winograd_tile_n;  // goes in-depth
    const size_t tile_m = winograd_tile_m;  // goes over flattened x and y

    const size_t input_tile_width = winograd_input_tile_width;
    const size_t input_tile_height = winograd_input_tile_height;

    const size_t nr_tiles_x =
        Align(arg.outputs[0].X().v, 4) / input_tile_width;  // input is already in winograd domain, so simply divide its
                                                        // width by tile's width to get tiles count
    const size_t nr_tiles_y = Align(arg.outputs[0].Y().v, 8) / input_tile_height;

    dispatchData.gws[0] = arg.outputs[0].Feature().v / tile_n;
    dispatchData.gws[1] = nr_tiles_x * nr_tiles_y / tile_m;
    dispatchData.gws[2] = input_tile_width * input_tile_height * arg.inputs[0].Batch().v;

    dispatchData.lws[0] = 8;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_Winograd_2x3_s1::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}

bool ConvolutionKernel_Winograd_2x3_s1::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        return false;
    }

    const convolution_params& cp = static_cast<const convolution_params&>(p);

    const bool bStrideOK = (cp.stride.x == 1 && cp.stride.y == 1);
    const bool bDilationOK = (cp.stride.x == 1 && cp.stride.y == 1);
    const bool bFilter3x3 = (cp.filterSize.x == 3 && cp.filterSize.y == 3);

    if (!bStrideOK || !bDilationOK || !bFilter3x3) {
        return false;
    }

    return true;
}

KernelsData ConvolutionKernel_Winograd_2x3_s1::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}
}  // namespace kernel_selector

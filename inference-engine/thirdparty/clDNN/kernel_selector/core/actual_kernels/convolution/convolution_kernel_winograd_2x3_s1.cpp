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
    k.EnableSubGroup();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    return k;
}

JitConstants ConvolutionKernel_Winograd_2x3_s1::GetJitConstants(const convolution_params& params,
                                                                const DispatchData& runInfo) const {
    JitConstants jit = Parent::GetJitConstants(params, runInfo);

    const size_t input_tile_width = winograd_input_tile_width;
    const size_t input_tile_height = winograd_input_tile_height;
    const size_t winograd_filter_height =
        params.filterSize.y;  // for this format, winograd filter is considered to be a set of 1d filters so its height
                              // should remain the same as original filter's

    const size_t nr_tiles_x =
        Align(params.output.X().v, 4) / input_tile_width;  // input is already in winograd domain, so simply divide its
                                                           // width by tile's width to get tiles count
    const size_t nr_tiles_y = Align(params.output.Y().v, 8) / input_tile_height;
    const size_t total_tiles_count = nr_tiles_x * nr_tiles_y;

    jit.AddConstants({
        MakeJitConstant("INPUT0_SIZE_WINOGRAD_X", Align(params.inputs[0].X().v, 4)),
        MakeJitConstant("INPUT0_SIZE_WINOGRAD_Y", Align(params.inputs[0].Y().v - 2, 8) + 2),
        MakeJitConstant("N", params.output.Feature().v),
        MakeJitConstant("M", total_tiles_count),
        MakeJitConstant("K", params.inputs[0].Feature().v * winograd_filter_height),
    });

    return jit;
}

ConvolutionKernel_Winograd_2x3_s1::Parent::DispatchData ConvolutionKernel_Winograd_2x3_s1::SetDefault(
    const convolution_params& arg,
    int) const {
    Parent::DispatchData runInfo = Parent::SetDefault(arg);

    const size_t tile_n = winograd_tile_n;  // goes in-depth
    const size_t tile_m = winograd_tile_m;  // goes over flattened x and y

    const size_t input_tile_width = winograd_input_tile_width;
    const size_t input_tile_height = winograd_input_tile_height;

    const size_t nr_tiles_x =
        Align(arg.output.X().v, 4) / input_tile_width;  // input is already in winograd domain, so simply divide its
                                                        // width by tile's width to get tiles count
    const size_t nr_tiles_y = Align(arg.output.Y().v, 8) / input_tile_height;

    runInfo.gws0 = arg.output.Feature().v / tile_n;
    runInfo.gws1 = nr_tiles_x * nr_tiles_y / tile_m;
    runInfo.gws2 = input_tile_width * input_tile_height * arg.inputs[0].Batch().v;

    runInfo.lws0 = 8;
    runInfo.lws1 = 1;
    runInfo.lws2 = 1;

    runInfo.efficiency = FORCE_PRIORITY_4;

    return runInfo;
}

bool ConvolutionKernel_Winograd_2x3_s1::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    const convolution_params& cp = static_cast<const convolution_params&>(p);

    const bool bStrideOK = (cp.stride.x == 1 && cp.stride.y == 1);
    const bool bDilationOK = (cp.stride.x == 1 && cp.stride.y == 1);
    const bool bFilter3x3 = (cp.filterSize.x == 3 && cp.filterSize.y == 3);
    const bool bNoSplit = cp.split == 1;

    if (!bStrideOK || !bDilationOK || !bFilter3x3 || !bNoSplit) {
        return false;
    }

    return true;
}

KernelsData ConvolutionKernel_Winograd_2x3_s1::GetKernelsData(const Params& params,
                                                              const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector
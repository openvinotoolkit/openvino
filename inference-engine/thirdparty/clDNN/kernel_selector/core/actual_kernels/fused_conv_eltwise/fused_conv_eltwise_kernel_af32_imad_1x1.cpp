/*
// Copyright (c) 2019-2020 Intel Corporation
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

#include "fused_conv_eltwise_kernel_af32_imad_1x1.h"

static size_t GetTileLength(size_t out_xy, size_t out_f, size_t min_threads) {
    for (int tile_len = 14; tile_len > 0; tile_len--) {
        // Kernel writes 32 output features per HW thread
        size_t threads = (out_xy / tile_len) * out_xy * out_f / 32;
        // Chose largest valid tile with enough HW threads
        if ((out_xy % tile_len == 0) && (threads >= min_threads)) {
            return tile_len;
        }
    }
    return out_xy % 8 ? (out_xy % 7 ? 1 : 7) : 8;
}

namespace kernel_selector {

ParamsKey fused_conv_eltwise_kernel_af32_imad_1x1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::byxf_af32);
    k.EnableOutputLayout(DataLayout::byxf_af32);
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.EnableInt8Quantization();
    k.EnableOutputCalibration();
    k.EnableFusedConvEltwInt8Quantization();
    k.EnableFusedConvEltwOutputCalibration();
    k.DisableTuning();
    k.EnableFusedConvEltwiseRWOutOpt();
    k.EnableEltwiseStride();
    return k;
}

bool fused_conv_eltwise_kernel_af32_imad_1x1::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    KernelData kd = KernelData::Default<fused_conv_eltwise_params>(p);
    fused_conv_eltwise_params& newParams = *static_cast<fused_conv_eltwise_params*>(kd.params.get());

    if (newParams.conv.filterSize.x != 1 || newParams.conv.filterSize.y != 1)
        return false;

    if (newParams.conv.padding.x != 0 || newParams.conv.padding.y != 0)
        return false;

    if (newParams.output.Feature().v % 32 != 0)
        return false;

    const auto& input = newParams.inputs[0];

    // we do not support padded input
    if (input.X().pad.Total() != 0 || input.Y().pad.Total() != 0)
        return false;

    if (newParams.conv.split != 1)
        return false;

    return true;
}

fused_conv_eltwise_kernel_base::DispatchData fused_conv_eltwise_kernel_af32_imad_1x1::SetDefault(
    const fused_conv_eltwise_params& arg,
    int) const {
    DispatchData runInfo = Parent::SetDefault(arg);

    // Sub-group size
    constexpr size_t sub_group_size = 8;

    const auto of_maps = arg.output.Feature().v;
    const size_t of_maps_per_batch = RoundUp(of_maps, 32);
    const size_t of_maps_total = of_maps_per_batch * arg.output.Batch().v;

    // Need to have at least 4 HW threads per EU
    const size_t tile_length = GetTileLength(arg.output.X().v, of_maps_total, arg.engineInfo.computeUnitsCount * 4);
    runInfo.cldnnStyle.blockWidth = tile_length;

    runInfo.efficiency = FORCE_PRIORITY_1;

    runInfo.gws0 = arg.output.X().v * arg.output.Y().v / tile_length;
    runInfo.gws1 = of_maps_total / 4;  // TILE_DEPTH==4
    runInfo.gws2 = 1;

    runInfo.lws0 = 1;
    runInfo.lws1 = sub_group_size;
    runInfo.lws2 = 1;

    return runInfo;
}

JitConstants fused_conv_eltwise_kernel_af32_imad_1x1::GetJitConstants(const fused_conv_eltwise_params& params,
                                                                      const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws1));

    jit.AddConstant(MakeJitConstant("TILE_LENGTH", runInfo.cldnnStyle.blockWidth));
    jit.AddConstant(MakeJitConstant("TILE_DEPTH", 4));

    if (params.non_conv_scale != 1.0f)
        jit.AddConstant(MakeJitConstant("NON_CONV_SCALE", params.non_conv_scale));

    jit.Merge(MakeActivationJitConstants(params.conv.activations, GetUnitType(params), "_CONV_TYPED", true));
    jit.Merge(MakeActivationJitConstants(params.activations,  GetUnitType(params), "_ELTW_TYPED", true));
    jit.Merge(MakeTypeJitConstants(Datatype::F32, "float"));

    return jit;
}

KernelsData fused_conv_eltwise_kernel_af32_imad_1x1::GetKernelsData(const Params& params,
                                                                    const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector

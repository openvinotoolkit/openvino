// Copyright (c) 2020 Intel Corporation
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

#include "convolution_kernel_b_fs_yx_fsv16_imad_3x3_ks.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>
#include <iostream>

//
// Kernel specific constants
//
#define SIMD_SIZE 16

static size_t getOutBlock_X(size_t output_size_x) {
    auto output_block_width = 7;
    if (output_size_x % 8 == 0)
        output_block_width = 8;
    return output_block_width;
}


namespace kernel_selector {

ParamsKey Convolution_kernel_b_fs_yx_fsv16_imad_3x3_ks::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);

    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.DisableTuning();
    return k;
}

KernelsData Convolution_kernel_b_fs_yx_fsv16_imad_3x3_ks::GetKernelsData(const Params& params,
                                                                      const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

JitConstants Convolution_kernel_b_fs_yx_fsv16_imad_3x3_ks::GetJitConstants(const convolution_params& params,
                                                                        const DispatchData& kd) const {
    auto mem_consts = Parent::GetJitConstants(params, kd);
    const auto& output = params.output;

    mem_consts.AddConstants({MakeJitConstant("OUT_BLOCK_WIDTH", getOutBlock_X(output.X().v))});

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = {"",
                                             {"out_b", "(out_f + get_sub_group_id() * 16)", "out_y", "out_x + i"},
                                             "dequantized",
                                             input_dt,
                                             1};
        conf_scalar.SetLoopAxes({ Tensor::DataChannelName::X }, true);
        mem_consts.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return mem_consts;
}  // GetJitConstants

ConvolutionKernelBase::DispatchData Convolution_kernel_b_fs_yx_fsv16_imad_3x3_ks::SetDefault(
    const convolution_params& params,
    int) const {
    DispatchData kd;
    const auto& output = params.output;

    auto output_block_width = getOutBlock_X(output.X().v);
    kd.gws0 = output.X().v / output_block_width;
    kd.gws1 = output.Y().v;
    kd.gws2 = output.Batch().v * output.Feature().v * 2;

    kd.lws0 = 1;
    kd.lws1 = 1;
    kd.lws2 = SIMD_SIZE * 4;

    kd.cldnnStyle = {0, 0, 0, 0, 0};
    kd.gemmStyle = {0, 0, 0, 0, 0, 0};

    kd.efficiency = FORCE_PRIORITY_1;

    return kd;
}  // SetDefault

bool Convolution_kernel_b_fs_yx_fsv16_imad_3x3_ks::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if (newParams.output.Feature().v % (2 * SIMD_SIZE) != 0) {
        return false;
    }

    if ((newParams.filterSize.x != newParams.filterSize.y) ||
        newParams.filterSize.x != 3) {
        // Fitler size needs to be 3x3
        return false;
    }

    if ((newParams.stride.x != newParams.stride.y) ||
        (newParams.stride.x != 1 && newParams.stride.x != 2)) {
        // Strides must be 1x1 or 2x2
        return false;
    }

    if (newParams.output.X().v % 8 != 0 && newParams.output.X().v % 7 != 0) {
        return false;
    }

    if (CeilDiv(newParams.inputs[0].Feature().v, 16) % 4 != 0) {
        return false;
    }

    const auto& output = newParams.output;
    auto output_block_width = getOutBlock_X(output.X().v);
    size_t eu_count = params.engineInfo.computeUnitsCount;
    auto global_size =
        (output.X().v / output_block_width) * output.Y().v * ((output.Batch().v * output.Feature().v));
    if ((global_size / 16) > (eu_count * 7)) {
        return false;
    }

    if (newParams.groups != 1 || newParams.split != 1)
        return false;

    return true;
}
}  // namespace kernel_selector

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


#include "convolution_kernel_b_fs_yx_fsv16_imad_3x3.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>
#include <iostream>

//
// Kernel specific constants
//
#define SIMD_SIZE 16

static size_t getOutBlock_X(const size_t output_size_x, const size_t stride_x, const size_t filter_size_x) {
    size_t output_block_width = 0;
    size_t max_block_size = std::min((SIMD_SIZE - filter_size_x) / stride_x + 1, output_size_x);

    if (output_size_x <= max_block_size)
        return output_size_x;

    for (size_t block = 4; block <= max_block_size; ++block) {
        if (output_size_x % block == 0)
            output_block_width = block;
    }
    if (output_block_width == 0 && output_size_x < max_block_size * 3) {
        size_t min_overhang = max_block_size;
        for (size_t block = 4; block <= max_block_size; ++block) {
            size_t overhang = block - output_size_x % block;
            if (overhang <= min_overhang) {
                min_overhang = overhang;
                output_block_width = block;
            }
        }
    }

    if (output_block_width == 0) {
        output_block_width = max_block_size;
    }
    return output_block_width;
}

static size_t get_ofm_per_wi(const size_t output_size_f) {
    if (output_size_f % 32 == 0)
        return 2;
    return 1;
}

namespace kernel_selector {

ParamsKey Convolution_kernel_b_fs_yx_fsv16_imad_3x3::GetSupportedKey() const {
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

KernelsData Convolution_kernel_b_fs_yx_fsv16_imad_3x3::GetKernelsData(const Params& params,
                                                                   const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

JitConstants Convolution_kernel_b_fs_yx_fsv16_imad_3x3::GetJitConstants(const convolution_params& params,
                                                                     const DispatchData& kd) const {
    auto mem_consts = Parent::GetJitConstants(params, kd);
    const auto& output = params.output;

    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_WIDTH", getOutBlock_X(output.X().v, params.stride.x, params.filterSize.x)));
    mem_consts.AddConstant(MakeJitConstant("OFM_BLOCKS_PER_SIMD", get_ofm_per_wi(output.Feature().v)));
    mem_consts.AddConstant(MakeJitConstant("OFM_SIZE_PER_SIMD", SIMD_SIZE * get_ofm_per_wi(output.Feature().v)));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = {"", {"out_b", "out_f + j * 16", "out_y", "out_x + i"}, "dequantized", input_dt, 1};
        conf_scalar.SetLoopAxes({ Tensor::DataChannelName::X }, true);
        mem_consts.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return mem_consts;
}  // GetJitConstants

ConvolutionKernelBase::DispatchData Convolution_kernel_b_fs_yx_fsv16_imad_3x3::SetDefault(const convolution_params& params,
                                                                           int) const {
    DispatchData kd;
    const auto& output = params.output;
    auto output_block_width = getOutBlock_X(output.X().v, params.stride.x, params.filterSize.x);
    auto ofm_blocks_per_simd = get_ofm_per_wi(output.Feature().v);

    kd.gws0 = CeilDiv(output.X().v, output_block_width);
    kd.gws1 = output.Y().v;
    kd.gws2 = output.Batch().v * Align(output.Feature().v / ofm_blocks_per_simd, SIMD_SIZE);

    kd.lws0 = 1;
    kd.lws1 = 1;
    kd.lws2 = SIMD_SIZE;

    kd.cldnnStyle = {0, 0, 0, 0, 0};
    kd.gemmStyle = {0, 0, 0, 0, 0, 0};

    if (params.filterSize.x == 3)
        kd.efficiency = FORCE_PRIORITY_2;
    else
        kd.efficiency = FORCE_PRIORITY_5;

    return kd;
}  // SetDefault

bool Convolution_kernel_b_fs_yx_fsv16_imad_3x3::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if ((newParams.filterSize.x != newParams.filterSize.y) ||
        (newParams.filterSize.x != 3 && newParams.filterSize.x != 5)) {
        // Fitler size needs to be 3x3 or 5x5
        return false;
    }

    if ((newParams.stride.x != newParams.stride.y) ||
        (newParams.stride.x != 1 && newParams.stride.x != 2)) {
        // Strides must be 1x1 or 2x2
        return false;
    }

    if (newParams.groups != 1 || newParams.split != 1)
        return false;

    return true;
}
}  // namespace kernel_selector

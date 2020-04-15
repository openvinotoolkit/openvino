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


#include "convolution_kernel_b_fs_yx_fsv16_imad_1x1.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>
#include <iostream>
#include <algorithm>

//
// Kernel specific constants
//
#define SIMD_SIZE 16

namespace kernel_selector {

namespace {

size_t getOutBlock_X(size_t output_size_x, size_t stride_x) {
    size_t output_block_width = 0;
    size_t max_block_size = std::min((SIMD_SIZE - 1) / stride_x + 1, output_size_x);

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

bool should_k_slice(const convolution_params& params, size_t output_block_width) {
    constexpr float preferred_eu_occupancy = 5.f;
    if (params.inputs[0].Feature().v % (16 * 4) != 0)
        return false;

    size_t eu_count = params.engineInfo.computeUnitsCount;
    auto global_size = CeilDiv(params.output.X().v, output_block_width) *
        params.output.Y().v *
        params.output.Batch().v * Align(CeilDiv(params.output.Feature().v, 2), SIMD_SIZE);
    auto threads = global_size / SIMD_SIZE;
    auto optimal_threads_num = eu_count * preferred_eu_occupancy;
    return threads < optimal_threads_num;
}

}  // namespace

Convolution_kernel_b_fs_yx_fsv16_imad_1x1::Convolution_kernel_b_fs_yx_fsv16_imad_1x1()
    : ConvolutionKernelBase("convolution_gpu_b_fs_yx_fsv16_imad_1x1") {
    for (size_t bw = 1; bw <= SIMD_SIZE; ++bw) {
        for (auto exe : ConvolutionKernelBase::autoTuneOptions) {
            all_tune_params.push_back(AutoTuneParams{ bw, true, exe });
            all_tune_params.push_back(AutoTuneParams{ bw, false, exe });
        }
    }
}

ParamsKey Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetSupportedKey() const {
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
    return k;
}

JitConstants Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetJitConstants(const convolution_params& params,
                                                                        const DispatchData& kd) const {
    auto mem_consts = Parent::GetJitConstants(params, kd);
    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_WIDTH", kd.cldnnStyle.blockWidth));
    mem_consts.AddConstant(MakeJitConstant("FEATURE_LWS_SPLIT", kd.cldnnStyle.prefetch));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = {"", {"out_b", "out_f + out_f_offset", "out_y", "out_x + i"}, "dequantized", input_dt, 1 };
        conf_scalar.SetLoopAxes({ Tensor::DataChannelName::X }, true);
        mem_consts.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return mem_consts;
}  // GetJitConstants

ConvolutionKernelBase::DispatchData Convolution_kernel_b_fs_yx_fsv16_imad_1x1::SetDefault(const convolution_params& params,
                                                                                          int index) const {
    DispatchData kd;
    const auto& output = params.output;
    auto tune_params = GetAutoTuneParams(params, index);
    size_t k_slices = tune_params.k_slicing ? 4 : 1;

    kd.gws0 = CeilDiv(output.X().v, tune_params.out_block_width);
    kd.gws1 = output.Y().v;
    kd.gws2 = output.Batch().v * Align(CeilDiv(output.Feature().v, 2), SIMD_SIZE) * k_slices;

    kd.lws0 = 1;
    kd.lws1 = 1;
    kd.lws2 = SIMD_SIZE * k_slices;

    kd.cldnnStyle = {0, 0, 0, 0, 0};
    kd.gemmStyle = {0, 0, 0, 0, 0, 0};

    kd.cldnnStyle.blockWidth = tune_params.out_block_width;
    kd.cldnnStyle.prefetch = k_slices;

    kd.efficiency = FORCE_PRIORITY_2;

    return kd;
}  // SetDefault

bool Convolution_kernel_b_fs_yx_fsv16_imad_1x1::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if ((newParams.filterSize.x != newParams.filterSize.y) ||
        newParams.filterSize.x != 1) {
        // Fitler size needs to be 1x1
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

Convolution_kernel_b_fs_yx_fsv16_imad_1x1::AutoTuneParams
Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetAutoTuneParams(const convolution_params& params, int index) const {
    if (index >= 0 && index < static_cast<int>(all_tune_params.size())) {
        return all_tune_params[index];
    }
    AutoTuneParams default_params;
    default_params.out_block_width = getOutBlock_X(params.output.X().v, params.stride.x);
    default_params.k_slicing = should_k_slice(params, default_params.out_block_width);
    default_params.exe_mode = DEFAULT;
    return default_params;
}

bool Convolution_kernel_b_fs_yx_fsv16_imad_1x1::ValidateAutoTuneParams(const convolution_params& params,
                                                                       const AutoTuneParams& tune_params) const {
    if (tune_params.k_slicing && params.inputs[0].Feature().v % (16 * 4) != 0)
        return false;

    size_t max_block_size = std::min(static_cast<size_t>((SIMD_SIZE - 1) / params.stride.x + 1), params.output.X().v);
    if (tune_params.out_block_width > max_block_size)
        return false;

    return true;
}

KernelsData Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetKernelsData(const Params& params,
                                                                      const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

KernelsData Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetTunedKernelsDataByIndex(const Params & params,
                                                                                  const optional_params & options,
                                                                                  int autoTuneIndex) const {
    auto conv_params = static_cast<const convolution_params&>(params);
    auto tune_params = GetAutoTuneParams(conv_params, autoTuneIndex);
    if (!ValidateAutoTuneParams(conv_params, tune_params))
        return {};
    return GetCommonKernelsData(params, options, tune_params.exe_mode, autoTuneIndex);
}

KernelsData Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetKernelsDataForAutoTune(const Params & params,
                                                                                 const optional_params & options) const {
    if (!Validate(params, options)) {
        return {};
    }
    auto& conv_params = static_cast<const convolution_params&>(params);

    KernelsData res = {};

    for (size_t i = 0; i < all_tune_params.size(); i++) {
        auto tune_params = GetAutoTuneParams(conv_params, static_cast<int>(i));
        if (!ValidateAutoTuneParams(conv_params, tune_params))
            continue;
        KernelsData kd = GetTunedKernelsDataByIndex(params, options, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

}  // namespace kernel_selector

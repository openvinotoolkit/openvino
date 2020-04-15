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

#include "convolution_kernel_b_fs_yx_fsv_16_32_imad_dw.hpp"

#include <vector>
#include <string>
#include <algorithm>

namespace kernel_selector {

ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw()
    : ConvolutionKernelBase("convolution_gpu_b_fs_yx_fsv_16_32_imad_dw") {
    std::vector<size_t> simd_sizes = { 8, 16 };
    std::vector<size_t> tile_x_sizes = { 1, 2, 3, 4, 5, 7, 8, 11, 16, 24, 32 };
    std::vector<std::string> exe_modes = ConvolutionKernelBase::autoTuneOptions;

    constexpr size_t max_block_size = 32 * 8;

    for (auto simd : simd_sizes) {
        for (size_t tile_x = 1; tile_x <= 32; ++tile_x) {
            if (simd * tile_x > max_block_size)
                continue;
            for (auto exe_mode : exe_modes) {
                all_tune_params.push_back(AutoTuneParams{ simd, tile_x, exe_mode });
            }
        }
    }
}

ParamsKey kernel_selector::ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_WEIGHTS);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);
    k.EnableDepthwiseSeparableOpt();
    k.EnableGroupedConvolution();
    return k;
}

bool ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    auto conv_params = static_cast<const convolution_params&>(params);

    if (conv_params.inputs[0].GetLayout() != conv_params.output.GetLayout())
        return false;

    if (conv_params.groups != conv_params.output.Feature().v || conv_params.groups != conv_params.inputs[0].Feature().v)
        return false;

    // Additional checks for asymmetric data
    if (conv_params.quantization == QuantizationType::ASYMMETRIC_DATA ||
        conv_params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS) {
        // Needs compensation optimization
        if (conv_params.compensation.empty())
            return false;
        // Padding not supported
        const auto inputLimitX = (conv_params.output.X().v - 1) * conv_params.stride.x
                               + (conv_params.filterSize.x - 1) * conv_params.dilation.x + 1;
        const auto inputLimitY = (conv_params.output.Y().v - 1) * conv_params.stride.y
                               + (conv_params.filterSize.y - 1) * conv_params.dilation.y + 1;
        const auto inputLimitZ = (conv_params.output.Z().v - 1) * conv_params.stride.z
                               + (conv_params.filterSize.z - 1) * conv_params.dilation.z + 1;

        bool needs_pad = false;
        needs_pad |= conv_params.padding.x != 0;
        needs_pad |= conv_params.padding.y != 0;
        needs_pad |= conv_params.padding.z != 0;
        needs_pad |= inputLimitX > conv_params.output.X().v;
        needs_pad |= inputLimitY > conv_params.output.Y().v;
        needs_pad |= inputLimitZ > conv_params.output.Z().v;

        if (needs_pad)
            return false;
    }

    return true;
}

WeightsLayout ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetPreferredWeightsLayout(const convolution_params& params) const {
    if (params.output.GetLayout() == DataLayout::b_fs_yx_fsv16)
        return WeightsLayout::gs_oi_yxs_gsv16_yxsv4;
    else
        return WeightsLayout::gs_oi_yxs_gsv32_yxsv4;
}

ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::AutoTuneParams
ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetAutoTuneParams(const convolution_params& params, int index) const {
    if (index >= 0 && index < static_cast<int>(all_tune_params.size())) {
        return all_tune_params[index];
    }
    AutoTuneParams tune_params;
    tune_params.simd = 16;
    if (params.output.GetLayout() == DataLayout::b_fs_yx_fsv16) {
        tune_params.tile_x = std::min((size_t)16, params.output.X().v);
    } else {
        tune_params.tile_x = std::min((size_t)8, params.output.X().v);
    }

    if (params.output.X().v < 3 * tune_params.tile_x && params.output.X().v % tune_params.tile_x != 0) {
        tune_params.tile_x = tune_params.tile_x / 2;
    }

    return tune_params;
}

ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::DispatchData
ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::SetDefault(const convolution_params& params, int autoTuneIndex) const {
    DispatchData kd;
    auto& out = params.output;

    auto tune_params = GetAutoTuneParams(params, autoTuneIndex);

    size_t fsv = 1;
    if (out.GetLayout() == DataLayout::b_fs_yx_fsv16) {
        fsv = 16;
    } else if (out.GetLayout() == DataLayout::b_fs_yx_fsv32) {
        fsv = 32;
    }

    std::vector<size_t> global = {
        CeilDiv(out.X().v, tune_params.tile_x),
        out.Y().v,
        CeilDiv(out.Feature().v, fsv) * tune_params.simd * out.Batch().v
    };
    std::vector<size_t> local = { 1, 1, tune_params.simd };

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    kd.gemmStyle = { 0, 0, 0, 0, 0, 0 };

    kd.cldnnStyle.blockWidth = tune_params.tile_x;

    kd.efficiency = params.stride.x == 1 ? FORCE_PRIORITY_1 : FORCE_PRIORITY_2;

    return kd;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetJitConstants(const convolution_params& params, const DispatchData& kd) const {
    auto mem_consts = Parent::GetJitConstants(params, kd);

    constexpr size_t imad_width = 4;
    auto filter_spatial = params.weights.X().v * params.weights.Y().v;
    auto filter_blocked = filter_spatial / imad_width * imad_width;

    mem_consts.AddConstant(MakeJitConstant("LWS0", kd.lws0));
    mem_consts.AddConstant(MakeJitConstant("LWS1", kd.lws1));
    mem_consts.AddConstant(MakeJitConstant("SIMD", kd.lws2));

    mem_consts.AddConstant(MakeJitConstant("TILE_X", kd.cldnnStyle.blockWidth));
    mem_consts.AddConstant(MakeJitConstant("FILTER_BLOCKED", filter_blocked));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        auto conf_1 = FusedOpsConfiguration("_1",
                                            { "b", "fused_ops_f", "y", "fused_ops_x" },
                                            "fused_ops_in",
                                            input_dt,
                                            1,
                                            LoadType::LT_ALIGNED_READ,
                                            BoundaryCheck::ENABLED,
                                            IndexType::TENSOR_COORD,
                                            Tensor::DataChannelName::FEATURE);
        auto conf_2 = conf_1;
        conf_2.suffix = "_2";
        conf_2.vec_size = 2;
        auto conf_4 = conf_1;
        conf_4.suffix = "_4";
        conf_4.vec_size = 4;
        mem_consts.Merge(MakeFusedOpsJitConstants(params, { conf_1, conf_2, conf_4 }));
    }

    return mem_consts;
}

KernelsData ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetTunedKernelsDataByIndex(const Params& params,
                                                                                const optional_params& options,
                                                                                int autoTuneIndex) const {
    auto convParams = static_cast<const convolution_params&>(params);
    auto tuneParams = GetAutoTuneParams(convParams, autoTuneIndex);
    return GetCommonKernelsData(params, options, tuneParams.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetKernelsDataForAutoTune(const Params& params,
                                                                               const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }
    auto& conv_params = static_cast<const convolution_params&>(params);

    KernelsData res = {};

    for (size_t i = 0; i < all_tune_params.size(); i++) {
        auto tune_params = GetAutoTuneParams(conv_params, static_cast<int>(i));
        KernelsData kd = GetTunedKernelsDataByIndex(params, options, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

}  // namespace kernel_selector

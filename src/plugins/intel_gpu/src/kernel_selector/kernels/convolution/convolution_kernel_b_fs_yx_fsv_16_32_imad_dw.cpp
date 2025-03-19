// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_b_fs_yx_fsv_16_32_imad_dw.hpp"

#include <vector>
#include <string>
#include <algorithm>

namespace kernel_selector {

ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw()
    : ConvolutionKernelBase("convolution_gpu_b_fs_yx_fsv_16_32_imad_dw") {
    std::vector<size_t> simd_sizes = { 8, 16 };
    std::vector<std::string> exe_modes = ConvolutionKernelBase::autoTuneOptions;

    // TODO: can be potentially improved for GPUs with support of LWS > 256
    constexpr size_t max_block_size = 32 * 8;
    constexpr size_t max_lws_size = 256;

    for (auto simd : simd_sizes) {
        for (size_t tile_x = 1; tile_x <= 32; ++tile_x) {
            if (simd * tile_x > max_block_size)
                continue;
            for (size_t lws0 = 1; lws0 <= 32; ++lws0) {
                for (size_t lws1 = 1; lws1 <= 32; ++lws1) {
                    if (lws0 * lws1 * simd > max_lws_size)
                        continue;
                    for (auto exe_mode : exe_modes) {
                        all_tune_params.push_back(AutoTuneParams{ simd, tile_x, lws0, lws1, false, exe_mode });
                        all_tune_params.push_back(AutoTuneParams{ simd, tile_x, lws0, lws1, true, exe_mode });
                    }
                }
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
    k.EnableGroupedConvolution();
    k.EnableDilation();
    k.EnableDynamicShapesSupport();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_blocked_read_write(); // for weights loading

    return k;
}

bool ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::Validate(const Params& params) const {
    if (!Parent::Validate(params))
        return false;

    auto conv_params = static_cast<const convolution_params&>(params);

    if (conv_params.inputs[0].GetLayout() != conv_params.outputs[0].GetLayout())
        return false;

    if (conv_params.inputs[0].Feature().is_dynamic || conv_params.outputs[0].Feature().is_dynamic)
        return false;

    if (conv_params.groups != conv_params.outputs[0].Feature().v || conv_params.groups != conv_params.inputs[0].Feature().v)
        return false;

    // Incorrect choose of TILE_X leads to accuracy issue
    if (conv_params.outputs[0].X().is_dynamic)
        return false;

    // For asymmetric data, kernel needs compensation optimization
    if (conv_params.compensation.empty() &&
        (conv_params.quantization == QuantizationType::ASYMMETRIC_DATA ||
         conv_params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS)) {
        return false;
    }

    return true;
}

WeightsLayout ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetPreferredWeightsLayout(const convolution_params& params) const {
    if (params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16)
        return WeightsLayout::gs_oi_yxs_gsv16_yxsv4;
    else
        return WeightsLayout::gs_oi_yxs_gsv32_yxsv4;
}

ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::AutoTuneParams
ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetAutoTuneParams(const convolution_params& params, int index) const {
    if (index >= 0 && index < static_cast<int>(all_tune_params.size())) {
        return all_tune_params[index];
    }

    auto& output = params.outputs[0];

    size_t fsv = 32;
    if (output.GetLayout() == DataLayout::b_fs_yx_fsv16) {
        fsv = 16;
    }
    size_t default_tile_x = fsv == 16 ? 16 : 8;

    AutoTuneParams tune_params;
    bool selected = false;
    // Helper function, selecting specified auto tune params if they are correct for current hw
    auto try_to_select = [&](size_t simd, size_t tile_x, size_t lws0, size_t lws1, bool preload_input_slm, std::string exe_mode) -> bool {
        tune_params.simd = simd;
        tune_params.tile_x = tile_x;
        tune_params.lws0 = lws0;
        tune_params.lws1 = lws1;
        tune_params.preload_input_slm = preload_input_slm;
        tune_params.exeMode = exe_mode;
        selected = ValidateAutoTuneParams(params, tune_params);
        return selected;
    };
    // List of fine-tuned known best cases
    bool filter_3x3 = params.filterSize.x == 3 && params.filterSize.y == 3;
    bool dilation_1x1 = params.dilation.x == 1 && params.dilation.y == 1;
    bool stride_1x1 = params.stride.x == 1 && params.stride.y == 1;
    bool stride_2x2 = params.stride.x == 2 && params.stride.y == 2;
    // Filter 3x3 with stride 1x1
    if (fsv == 16 && filter_3x3 && stride_1x1 && dilation_1x1 &&
        !output.X().is_dynamic && output.X().v == 75 &&
        !output.Y().is_dynamic && output.Y().v == 75) {
        try_to_select(16, 15, 1, 4, true, EXE_MODE_DEFAULT);
    }

    // Filter 3x3 with stride 2x2
    if (fsv == 16 && filter_3x3 && stride_2x2 && dilation_1x1 &&
        !output.X().is_dynamic && output.X().v == 75 &&
        !output.Y().is_dynamic && output.Y().v == 75) {
        try_to_select(16, 15, 1, 16, true, EXE_MODE_DEFAULT);
    }

    // Check if SLM can provide data reuse for current parameters
    bool use_slm_x = (params.filterSize.x - 1) * params.dilation.x + 1 >= params.stride.x;
    bool use_slm_y = (params.filterSize.y - 1) * params.dilation.y + 1 >= params.stride.y;
    // Small spatials are inefficent with SLM due to additional overhead
    bool use_slm_size = !output.X().is_dynamic && output.X().v >= default_tile_x &&
                        !output.Y().is_dynamic && output.Y().v >= default_tile_x;

    if (!selected && use_slm_y && use_slm_x && use_slm_size) {
        size_t tile_x = default_tile_x / 2;
        size_t lws1 = 4;
        size_t lws0 = 1;
        // First try to select optimal group size in y dimension as it provides most data reuse
        for (size_t c_lws1 = 4; c_lws1 <= 8; ++c_lws1) {
            if (Pad(output.Y().v, c_lws1) < Pad(output.Y().v, lws1))
                lws1 = c_lws1;
            else if (Pad(output.Y().v, c_lws1) == Pad(output.Y().v, lws1) && c_lws1 % 2 == 0)
                lws1 = c_lws1;
        }
        // For best hw utilization work-group size should be multiple of 2, so if y isn't force it in x
        if (lws1 % 2 != 0) {
            lws0 = 2;
        }
        // Local memory allocation works at 1kb granularity
        auto calc_slm_size_kb = [&](size_t tile_x, size_t lws0, size_t lws1) {
            size_t slm_x = (tile_x * lws0 - 1) * params.stride.x + (params.filterSize.x - 1) * params.dilation.x;
            size_t slm_y = (lws1 == 1) ? params.filterSize.x
                                       : (lws1 - 1) * params.stride.y + (params.filterSize.y - 1) * params.dilation.y + 1;
            return CeilDiv(slm_x * slm_y * fsv, 1024);
        };
        // Try to select optimal tile size with SLM and lws restrictions
        bool tile_selected = false;
        for (size_t c_tile = default_tile_x / 2; c_tile <= default_tile_x; ++c_tile) {
            bool c_better_pad = Pad(output.X().v, c_tile) <= Pad(output.X().v, tile_x);
            bool c_lws_tile = CeilDiv(output.X().v, c_tile) % lws0 == 0;
            // Compare SLM size with threads number to make sure occupancy isn't affected by slm usage
            bool c_slm_size = calc_slm_size_kb(c_tile, lws0, lws1) <= lws0 * lws1;
            if (c_better_pad && c_lws_tile && c_slm_size) {
                tile_x = c_tile;
                tile_selected = true;
            }
        }

        if (tile_selected && CeilDiv(output.X().v, tile_x) == 2)
            lws0 = 2;

        if (tile_selected)
            try_to_select(16, tile_x, lws0, lws1, true, EXE_MODE_DEFAULT);
    }

    if (!selected) {
        // Falback to non SLM path
        tune_params.simd = 16;
        tune_params.tile_x = 1;

        if (!output.X().is_dynamic) {
            tune_params.tile_x = std::min(default_tile_x, output.X().v);

            if (output.X().v < 3 * tune_params.tile_x && output.X().v % tune_params.tile_x != 0) {
                tune_params.tile_x = tune_params.tile_x / 2;
            }
        }

        tune_params.lws0 = 1;
        tune_params.lws1 = 1;
        tune_params.preload_input_slm = false;
        tune_params.exeMode = EXE_MODE_DEFAULT;
    }

    return tune_params;
}

bool ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::ValidateAutoTuneParams(const convolution_params& params,
                                                                         const AutoTuneParams& tparams) const {
    bool valid_tune_params = true;

    if (!IsSIMDSizeSupported(params.engineInfo, tparams.simd))
        return false;

    auto total_lws = tparams.simd * tparams.lws0 * tparams.lws1;
    valid_tune_params &= total_lws <= params.engineInfo.maxWorkGroupSize;

    size_t slm_preload_tile_x = (tparams.tile_x * tparams.lws0 - 1) * params.stride.x + (params.filterSize.x - 1) * params.dilation.x + 1;
    size_t slm_preload_tile_y = (tparams.lws1 == 1)
                                ? params.filterSize.y
                                : (tparams.lws1 - 1) * params.stride.y + (params.filterSize.y - 1) * params.dilation.y + 1;
    size_t fsv = params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 ? 16 : 32;
    auto total_slm = tparams.preload_input_slm ? slm_preload_tile_x * slm_preload_tile_y * fsv : 0;
    valid_tune_params &= total_slm <= params.engineInfo.maxLocalMemSize;

    // Check that tune params don't use needlesly many work-groups in x/y
    if (!params.outputs[0].X().is_dynamic) {
        valid_tune_params &= tparams.tile_x <= params.outputs[0].X().v;
        valid_tune_params &= tparams.tile_x * tparams.lws0 <= Align(params.outputs[0].X().v, 2);
    }
    if (!params.outputs[0].Y().is_dynamic) {
        valid_tune_params &= tparams.lws1 <= Align(params.outputs[0].Y().v, 2);
    }

    // Filter out combinations that are known to be sub-optimal in order to reduce search space
    valid_tune_params &= tparams.exeMode == EXE_MODE_DEFAULT;
    valid_tune_params &= tparams.preload_input_slm || tparams.lws0 * tparams.lws1 == 1;
    valid_tune_params &= !tparams.preload_input_slm || (tparams.lws0 * tparams.lws1) % 2 == 0;

    return valid_tune_params;
}

ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::DispatchData
ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::SetDefault(const convolution_params& params, int autoTuneIndex) const {
    DispatchData dispatchData;
    auto& out = params.outputs[0];

    auto tune_params = GetAutoTuneParams(params, autoTuneIndex);

    size_t fsv = 1;
    if (out.GetLayout() == DataLayout::b_fs_yx_fsv16) {
        fsv = 16;
    } else if (out.GetLayout() == DataLayout::b_fs_yx_fsv32) {
        fsv = 32;
    }

    dispatchData.gws = {
        Align(CeilDiv(out.X().v, tune_params.tile_x), tune_params.lws0),
        Align(out.Y().v, tune_params.lws1),
        CeilDiv(out.Feature().v, fsv) * tune_params.simd * out.Batch().v
    };
    dispatchData.lws = { tune_params.lws0, tune_params.lws1, tune_params.simd };

    dispatchData.gemmStyle = { 0, 0, 0, 0, 0, 0 };

    dispatchData.cldnnStyle.blockWidth = tune_params.tile_x;
    dispatchData.cldnnStyle.prefetch = tune_params.preload_input_slm;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const convolution_params&>(params);

    return p.stride.x == 1 ? FORCE_PRIORITY_1 : FORCE_PRIORITY_2;
}

bool ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::HasPaddedInput(const convolution_params& params) const {
    if (params.outputs[0].X().is_dynamic ||
        params.outputs[0].Y().is_dynamic ||
        params.outputs[0].Z().is_dynamic) {
        return false;
    }

    const auto inputLimitX = (params.outputs[0].X().v - 1) * params.stride.x
        + (params.filterSize.x - 1) * params.dilation.x + 1;
    const auto inputLimitY = (params.outputs[0].Y().v - 1) * params.stride.y
        + (params.filterSize.y - 1) * params.dilation.y + 1;
    const auto inputLimitZ = (params.outputs[0].Z().v - 1) * params.stride.z
        + (params.filterSize.z - 1) * params.dilation.z + 1;

    bool has_pad = true;
    has_pad &= params.padding_begin.x <= params.inputs[0].X().pad.before;
    has_pad &= params.padding_begin.y <= params.inputs[0].Y().pad.before;
    has_pad &= params.padding_begin.z <= params.inputs[0].Z().pad.before;
    has_pad &= inputLimitX <= params.padding_begin.x + params.inputs[0].X().v + params.inputs[0].X().pad.after;
    has_pad &= inputLimitY <= params.padding_begin.y + params.inputs[0].Y().v + params.inputs[0].Y().pad.after;
    has_pad &= inputLimitZ <= params.padding_begin.z + params.inputs[0].Z().v + params.inputs[0].Z().pad.after;

    return has_pad;
}

bool ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::ParamsHavePadding(const convolution_params& params) const {
    if (params.outputs[0].X().is_dynamic ||
        params.outputs[0].Y().is_dynamic ||
        params.outputs[0].Z().is_dynamic) {
        return true;
    }

    const auto inputLimitX = (params.outputs[0].X().v - 1) * params.stride.x
        + (params.filterSize.x - 1) * params.dilation.x + 1;
    const auto inputLimitY = (params.outputs[0].Y().v - 1) * params.stride.y
        + (params.filterSize.y - 1) * params.dilation.y + 1;
    const auto inputLimitZ = (params.outputs[0].Z().v - 1) * params.stride.z
        + (params.filterSize.z - 1) * params.dilation.z + 1;

    bool needs_pad = false;
    needs_pad |= params.padding_begin.x != 0;
    needs_pad |= params.padding_begin.y != 0;
    needs_pad |= params.padding_begin.z != 0;
    needs_pad |= inputLimitX > params.inputs[0].X().v;
    needs_pad |= inputLimitY > params.inputs[0].Y().v;
    needs_pad |= inputLimitZ > params.inputs[0].Z().v;

    return needs_pad;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const {
    auto mem_consts = Parent::GetJitConstants(params, dispatchData);

    constexpr size_t imad_width = 4;
    auto filter_spatial = params.weights.X().v * params.weights.Y().v;
    auto filter_blocked = filter_spatial / imad_width * imad_width;

    mem_consts.AddConstant(MakeJitConstant("LWS0", dispatchData.lws[0]));
    mem_consts.AddConstant(MakeJitConstant("LWS1", dispatchData.lws[1]));
    mem_consts.AddConstant(MakeJitConstant("SIMD", dispatchData.lws[2]));

    mem_consts.AddConstant(MakeJitConstant("TILE_X", dispatchData.cldnnStyle.blockWidth));
    mem_consts.AddConstant(MakeJitConstant("FILTER_BLOCKED", filter_blocked));
    mem_consts.AddConstant(MakeJitConstant("PRELOAD_INPUT_TO_SLM", dispatchData.cldnnStyle.prefetch));

    auto needs_boundary_check = ParamsHavePadding(params) &&
        (!HasPaddedInput(params) ||
         params.quantization == QuantizationType::ASYMMETRIC_DATA ||
         params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);
    mem_consts.AddConstant(MakeJitConstant("CHECK_BOUNDARY", needs_boundary_check));

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
                                                                                    int autoTuneIndex) const {
    auto convParams = static_cast<const convolution_params&>(params);
    auto tuneParams = GetAutoTuneParams(convParams, autoTuneIndex);
    if (!ValidateAutoTuneParams(convParams, tuneParams))
        return {};
    return GetCommonKernelsData(params, tuneParams.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetKernelsDataForAutoTune(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < all_tune_params.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

void ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw::GetUpdateDispatchDataFunc(KernelData& kd) const {
    Parent::GetUpdateDispatchDataFunc(kd);
}

}  // namespace kernel_selector

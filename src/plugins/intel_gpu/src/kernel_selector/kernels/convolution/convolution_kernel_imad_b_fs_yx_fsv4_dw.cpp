// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_imad_b_fs_yx_fsv4_dw.hpp"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>
#include <string>
#include <algorithm>

namespace kernel_selector {

namespace {
constexpr size_t fsv = 4;
constexpr size_t max_reg_usage = 64;

enum mode : size_t {
    naive = 0,
    preload_input = 1,
    preload_weights = 2,
    tiled = 4
};

}  // namespace

ConvolutionKernel_imad_b_fs_yx_fsv4_dw::ConvolutionKernel_imad_b_fs_yx_fsv4_dw()
    : ConvolutionKernelBase("convolution_gpu_b_fs_yx_fsv4_dw") {

    std::vector<size_t> simd_sizes = { 8, 16 };
    std::vector<size_t> tile_y_sizes = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::vector<std::string> exe_modes = ConvolutionKernelBase::autoTuneOptions;

    for (auto& simd : simd_sizes) {
        for (auto& ty : tile_y_sizes) {
            for (size_t tx = 1; tx <= simd; ++tx) {
                for (auto& exec : exe_modes) {
                    all_tune_params.push_back(AutoTuneParams{ tx, ty, simd, true, true, true, exec });
                }
            }
        }
    }

    for (size_t ox = 1; ox < 17; ++ox) {
        for (auto& exec : exe_modes) {
            all_tune_params.push_back(AutoTuneParams{ ox, 1, 1, false, true, true, exec });
            all_tune_params.push_back(AutoTuneParams{ ox, 1, 1, false, true, false, exec });
        }
    }

    for (size_t ox = 1; ox < 17; ++ox) {
        for (auto& exec : exe_modes) {
            all_tune_params.push_back(AutoTuneParams{ ox, 1, 1, false, false, false, exec });
            all_tune_params.push_back(AutoTuneParams{ ox, 1, 1, false, false, true, exec });
        }
    }
}

ParamsKey ConvolutionKernel_imad_b_fs_yx_fsv4_dw::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableGroupedConvolution();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_imad_b_fs_yx_fsv4_dw::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_shuffle();

    return k;
}

bool ConvolutionKernel_imad_b_fs_yx_fsv4_dw::Validate(const Params& params) const {
    if (!Parent::Validate(params)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if (newParams.inputs[0].Feature().v != newParams.groups || newParams.outputs[0].Feature().v != newParams.groups)
        return false;

    if (newParams.outputs[0].Feature().pad.before % fsv != 0)
        return false;

    return true;
}

bool ConvolutionKernel_imad_b_fs_yx_fsv4_dw::ValidateAutoTuneParams(const convolution_params& params, const AutoTuneParams& tune_params) const {
    // Checks that tune_params can be used for specified convolution_params
    auto& weights = params.weights;

    if (tune_params.tiled) {
        bool tiled_x_once = tune_params.tiled_simd >= (weights.X().v - 1) * params.dilation.x + 1;
        if (!tiled_x_once)
            return false;

        auto max_tile_x = (tune_params.tiled_simd - 1 - (weights.X().v - 1) * params.dilation.x) / params.stride.x + 1;
        if (tune_params.block_x > max_tile_x)
            return false;

        if (tune_params.block_y != 1 && params.stride.y != params.dilation.y)
            return false;

        if (tune_params.block_y > params.outputs[0].Y().v)
            return false;
    } else if (tune_params.preload_input) {
        auto line_size = (tune_params.block_x - 1) * params.stride.x + (weights.X().v - 1) * params.dilation.x + 1;

        size_t reg_usage = tune_params.block_x * 4
                         + line_size * weights.Y().v
                         + Align(weights.X().v * weights.Y().v, 4) * tune_params.preload_weights;
        if (reg_usage > max_reg_usage)
            return false;

        if (tune_params.block_y != 1 && params.stride.y != params.dilation.y)
            return false;

        if (tune_params.block_y > params.outputs[0].Y().v)
            return false;
    } else {
        size_t block_size = tune_params.block_x * 4 + Align(weights.X().v * weights.Y().v, 4) * tune_params.preload_weights;
        if (block_size > max_reg_usage)
            return false;
    }

    return true;
}

ConvolutionKernel_imad_b_fs_yx_fsv4_dw::AutoTuneParams ConvolutionKernel_imad_b_fs_yx_fsv4_dw::GetAutoTuneParams(const convolution_params& params,
                                                                                                                 int index) const {
    if (index >= 0 && index < static_cast<int>(all_tune_params.size())) {
        return all_tune_params[index];
    }

    auto& output = params.outputs[0];
    auto& weights = params.weights;

    AutoTuneParams tune_params;

    bool is_1_by_x = weights.X().v == 1;

    // Check that we can preload x and calculate at least two output values
    constexpr size_t min_preload_width = 2;
    size_t min_preload_regs = ((min_preload_width - 1) * params.stride.x + (weights.X().v - 1) * params.dilation.x + 1) * params.weights.Y().v
                            + min_preload_width * 4
                            + Align(weights.X().v * weights.Y().v, 4) <= max_reg_usage;
    bool can_preload_input = min_preload_regs <= max_reg_usage && !is_1_by_x;

    if (can_preload_input) {
        // Find best block size:
        // 1. Try to make gws spatial divisible by 8
        // 2. Registers don't spill (less or equal to max_reg_usage)
        // 3. Maximize block size
        // 4. Minimize wasted output x (overhang)
        constexpr size_t pref_spatial_multiple = 8;
        size_t best_x_1 = 2;
        size_t best_x_8 = 0;
        size_t best_blocks_x_1 = output.X().v;
        size_t best_blocks_x_8 = output.X().v;
        size_t best_overhang_x_1 = 0;
        size_t best_overhang_x_8 = 0;
        for (size_t x = 2; x < 17; ++x) {
            size_t line_size = (x - 1) * params.stride.x + (weights.X().v - 1) * params.dilation.x + 1;
            size_t reg_usage = x * 4
                             + line_size * weights.Y().v
                             + Align(weights.X().v * weights.Y().v, 4);
            if (x > output.X().v)
                break;
            if (reg_usage > max_reg_usage)
                break;
            size_t blocks_x = CeilDiv(output.X().v, x);
            size_t overhang = blocks_x * x - output.X().v;

            if (blocks_x < best_blocks_x_1 || overhang < best_overhang_x_1) {
                best_x_1 = x;
                best_blocks_x_1 = blocks_x;
                best_overhang_x_1 = overhang;
            }
            if (blocks_x < best_blocks_x_8 || overhang < best_overhang_x_8) {
                if (blocks_x % pref_spatial_multiple == 0 ||
                    (blocks_x == 4 && output.Y().v % 2 == 0) ||
                    (blocks_x == 2 && output.Y().v % 4 == 0) ||
                    (blocks_x == 1 && output.Y().v % 8 == 0)) {
                    best_x_8 = x;
                    best_blocks_x_8 = blocks_x;
                    best_overhang_x_8 = overhang;
                }
            }
        }
        tune_params.tiled = false;
        tune_params.block_x = best_x_8 != 0 ? best_x_8 : best_x_1;
        tune_params.block_y = 1;
        tune_params.preload_input = true;
        tune_params.preload_weights = true;
        tune_params.tiled_simd = 0;
    } else {
        // Most basic path
        // 1. Try to make gws spatial divisible by 8
        // 2. Registers don't spill (less or equal to max_reg_usage)
        // 3. Maximize block size
        // 4. Minimize wasted output x (overhang)
        constexpr size_t pref_spatial_multiple = 8;
        size_t best_x_1 = 1;
        size_t best_x_8 = 0;
        size_t best_blocks_x_1 = output.X().v;
        size_t best_blocks_x_8 = output.X().v;
        size_t best_overhang_x_1 = 0;
        size_t best_overhang_x_8 = 0;
        for (size_t x = 1; x < 17; ++x) {
            if (x > output.X().v)
                break;
            size_t blocks_x = CeilDiv(output.X().v, x);
            size_t overhang = blocks_x * x - output.X().v;

            if (blocks_x < best_blocks_x_1 || overhang < best_overhang_x_1) {
                best_x_1 = x;
                best_blocks_x_1 = blocks_x;
                best_overhang_x_1 = overhang;
            }
            if (blocks_x < best_blocks_x_8 || overhang < best_overhang_x_8) {
                if (blocks_x % pref_spatial_multiple == 0 ||
                    (blocks_x == 4 && output.Y().v % 2 == 0) ||
                    (blocks_x == 2 && output.Y().v % 4 == 0) ||
                    (blocks_x == 1 && output.Y().v % 8 == 0)) {
                    best_x_8 = x;
                    best_blocks_x_8 = blocks_x;
                    best_overhang_x_8 = overhang;
                }
            }
        }
        tune_params.tiled = false;
        tune_params.block_x = best_x_8 != 0 ? best_x_8 : best_x_1;
        tune_params.block_y = 1;
        tune_params.preload_input = false;
        if (tune_params.block_x > 1 && (tune_params.block_x * 4 + Align(weights.X().v * weights.Y().v, 4)) <= max_reg_usage) {
            tune_params.preload_weights = true;
        } else {
            tune_params.preload_weights = false;
        }
        tune_params.tiled_simd = 0;
    }

    return tune_params;
}

JitConstants ConvolutionKernel_imad_b_fs_yx_fsv4_dw::GetJitConstants(const convolution_params& params,
                                                                     const DispatchData& dispatchData) const {
    auto mem_consts = Parent::GetJitConstants(params, dispatchData);

    size_t filter_block_size = 4;
    size_t min_blocked_leftovers = 4;

    auto filter_spatial = params.weights.X().v * params.weights.Y().v;

    auto filter_blocked = filter_spatial / filter_block_size * filter_block_size;
    auto filter_leftovers = filter_spatial - filter_blocked;
    if (filter_leftovers >= min_blocked_leftovers) {
        filter_blocked += filter_leftovers;
    }
    mem_consts.AddConstant(MakeJitConstant("FILTER_BLOCKED", filter_blocked));

    auto& work_mode = dispatchData.cldnnStyle.prefetch;
    bool tiled = (work_mode & mode::tiled) != 0;
    bool preload_input = (work_mode & mode::preload_input) != 0;
    bool preload_weights = (work_mode & mode::preload_weights) != 0;
    size_t simd = 16;
    size_t tile_x;
    size_t tile_y;
    size_t input_line_size;
    size_t output_block_x;

    if (tiled) {
        preload_weights = true;
        simd = dispatchData.lws[0];
        tile_x = dispatchData.cldnnStyle.blockWidth;
        tile_y = dispatchData.cldnnStyle.blockHeight;
        input_line_size = 1;
        output_block_x = 1;
    } else if (preload_input) {
        tile_x = 1;
        tile_y = dispatchData.cldnnStyle.blockHeight;
        output_block_x = dispatchData.cldnnStyle.blockWidth;
        input_line_size = (output_block_x - 1) * params.stride.x + (params.weights.X().v - 1) * params.dilation.x + 1;
    } else {
        tile_x = 1;
        tile_y = 1;
        input_line_size = 1;
        output_block_x = dispatchData.cldnnStyle.blockWidth;
    }

    mem_consts.AddConstant(MakeJitConstant("TILED", tiled));
    mem_consts.AddConstant(MakeJitConstant("PRELOAD_INPUT", preload_input));
    mem_consts.AddConstant(MakeJitConstant("PRELOAD_WEIGHTS", preload_weights));
    mem_consts.AddConstant(MakeJitConstant("SIMD", simd));
    mem_consts.AddConstant(MakeJitConstant("TILE_X", tile_x));
    mem_consts.AddConstant(MakeJitConstant("TILE_Y", tile_y));
    mem_consts.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));
    mem_consts.AddConstant(MakeJitConstant("OUTPUT_BLOCK_X", output_block_x));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        auto conf = FusedOpsConfiguration("",
                                          { "b", "f", "y", "(x + oxi + tile_x)" },
                                          "dequantized",
                                          input_dt,
                                          4,
                                          LoadType::LT_UNALIGNED,
                                          BoundaryCheck::ENABLED,
                                          IndexType::TENSOR_COORD,
                                          Tensor::DataChannelName::FEATURE);
        mem_consts.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return mem_consts;
}  // GetJitConstants

ConvolutionKernelBase::DispatchData ConvolutionKernel_imad_b_fs_yx_fsv4_dw::SetDefault(const convolution_params& params,
                                                                                       int autoTuneIndex) const {
    DispatchData dispatchData;
    auto& out = params.outputs[0];

    auto autoTuneParam = GetAutoTuneParams(params, autoTuneIndex);

    size_t global_x = CeilDiv(out.X().v, autoTuneParam.block_x);
    size_t global_y = CeilDiv(out.Y().v, autoTuneParam.block_y);

    if (autoTuneParam.tiled) {
        global_x = global_x * autoTuneParam.tiled_simd;
    }

    dispatchData.gws = { global_x, global_y, CeilDiv(out.Feature().v, fsv) * out.Batch().v };
    dispatchData.lws = { 1, 1, 1 };

    if (autoTuneParam.tiled) {
        dispatchData.lws[0] = autoTuneParam.tiled_simd;
    } else {
        auto in_layout = params.inputs[0].GetLayout();
        auto out_layout = params.outputs[0].GetLayout();
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X },
                                                                         { Tensor::DataChannelName::Y },
                                                                         { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};

        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    }

    dispatchData.gemmStyle = { 0, 0, 0, 0, 0, 0 };

    dispatchData.cldnnStyle.blockWidth = autoTuneParam.block_x;
    dispatchData.cldnnStyle.blockHeight = autoTuneParam.block_y;
    dispatchData.cldnnStyle.prefetch = (static_cast<size_t>(autoTuneParam.tiled) * mode::tiled)
                                     | (static_cast<size_t>(autoTuneParam.preload_input) * mode::preload_input)
                                     | (static_cast<size_t>(autoTuneParam.preload_weights) * mode::preload_weights);

    return dispatchData;
}  // SetDefault

KernelsPriority ConvolutionKernel_imad_b_fs_yx_fsv4_dw::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

KernelsData ConvolutionKernel_imad_b_fs_yx_fsv4_dw::GetTunedKernelsDataByIndex(const Params& params,
                                                                               int autoTuneIndex) const {
    auto convParams = static_cast<const convolution_params&>(params);
    auto tuneParams = GetAutoTuneParams(convParams, autoTuneIndex);
    return GetCommonKernelsData(params, tuneParams.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_imad_b_fs_yx_fsv4_dw::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

KernelsData ConvolutionKernel_imad_b_fs_yx_fsv4_dw::GetKernelsDataForAutoTune(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }
    auto& conv_params = static_cast<const convolution_params&>(params);

    KernelsData res = {};

    for (size_t i = 0; i < all_tune_params.size(); i++) {
        auto tune_params = GetAutoTuneParams(conv_params, static_cast<int>(i));
        if (!ValidateAutoTuneParams(conv_params, tune_params))
            continue;
        KernelsData kd = GetTunedKernelsDataByIndex(params, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

}  // namespace kernel_selector

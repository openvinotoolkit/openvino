// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_b_fs_yx_fsv16_imad_1x1.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>

//
// Kernel specific constants
//
static constexpr size_t fsv = 16;
static constexpr size_t simd = 16;

namespace kernel_selector {

Convolution_kernel_b_fs_yx_fsv16_imad_1x1::Convolution_kernel_b_fs_yx_fsv16_imad_1x1()
    : ConvolutionKernelBase("convolution_gpu_b_fs_yx_fsv16_imad_1x1") {
    // TODO: can be potentially improved for GPUs with support of LWS > 256
    constexpr size_t max_block_elements = 32;
    for (size_t bs = 1; bs <= 2 * simd; ++bs) {
        for (size_t bf = 1; bf <= 4; ++bf) {
            if (bs * bf > max_block_elements)
                continue;
            for (size_t split = 1; split <= 8; ++split) {
                if (bf > split)
                    continue;
                for (auto exe : ConvolutionKernelBase::autoTuneOptions) {
                    all_tune_params.push_back(AutoTuneParams{ bs, bf, split, exe });
                }
            }
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
    k.EnableInputWeightsType(WeightsType::UINT8);

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
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_WEIGHTS);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);
    return k;
}

DeviceFeaturesKey Convolution_kernel_b_fs_yx_fsv16_imad_1x1::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

JitConstants Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetJitConstants(const convolution_params& params,
                                                                        const DispatchData& dispatchData) const {
    auto mem_consts = Parent::GetJitConstants(params, dispatchData);
    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_SPATIAL", dispatchData.cldnnStyle.blockWidth));
    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_FEATURES", dispatchData.cldnnStyle.blockHeight));
    mem_consts.AddConstant(MakeJitConstant("FEATURE_SLM_SPLIT", dispatchData.cldnnStyle.prefetch));
    mem_consts.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    mem_consts.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        std::vector<std::string> idx_order = { "out_b",
                                               "(out_f + ofb * SIMD)",
                                               "_sub_group_shuffle(out_y_shuffle[os / SIMD], os % SIMD)",
                                               "_sub_group_shuffle(out_x_shuffle[os / SIMD], os % SIMD)" };
        FusedOpsConfiguration conf_scalar = {"_SCALAR",
                                             idx_order,
                                             "dequantized[ofb][os]",
                                             input_dt,
                                             1,
                                             LoadType::LT_UNALIGNED,
                                             BoundaryCheck::DISABLED };
        conf_scalar.SetLoopAxes({ Tensor::DataChannelName::X, Tensor::DataChannelName::Y }, true);
        mem_consts.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return mem_consts;
}  // GetJitConstants

ConvolutionKernelBase::DispatchData Convolution_kernel_b_fs_yx_fsv16_imad_1x1::SetDefault(const convolution_params& params,
                                                                                          int index) const {
    DispatchData dispatchData;
    const auto& output = params.outputs[0];
    auto tune_params = GetAutoTuneParams(params, index);
    size_t k_slices = tune_params.feature_slm_split;

    dispatchData.gws[0] = CeilDiv(output.X().v * output.Y().v, tune_params.out_block_spatial);
    dispatchData.gws[1] = CeilDiv(output.Feature().v, tune_params.out_block_features * simd) * simd * k_slices;
    dispatchData.gws[2] = output.Batch().v;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = simd * k_slices;
    dispatchData.lws[2] = 1;

    dispatchData.cldnnStyle = {0, 0, 0, 0, 0};
    dispatchData.gemmStyle = {0, 0, 0, 0, 0, 0};

    dispatchData.cldnnStyle.blockWidth = tune_params.out_block_spatial;
    dispatchData.cldnnStyle.blockHeight = tune_params.out_block_features;
    dispatchData.cldnnStyle.prefetch = k_slices;

    return dispatchData;
}  // SetDefault

KernelsPriority Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const convolution_params&>(params);

    const auto& output = p.outputs[0];
    auto tune_params = GetAutoTuneParams(p, -1);

    auto priority = FORCE_PRIORITY_2;

    auto in_f = p.weights.IFM().v;
    auto out_f = p.weights.OFM().v;
    auto batch = output.Batch().v;
    auto out_x = output.X().v;
    auto out_y = output.Y().v;

    bool x_strided = p.stride.x != 1;
    bool general_is_faster = false;

    // This kernel cannot split for large x, but general could
    general_is_faster |= CeilDiv(in_f, fsv) % 4 == 0
                         && (out_x % 15 == 0 || out_x % 16 == 0)
                         && tune_params.feature_slm_split == 1
                         && tune_params.out_block_spatial <= 8;

    // List of known cases where general kernel is better
    general_is_faster |= in_f == 24 && out_f == 144 && out_x == 75 && out_y == 75 && batch == 1;
    general_is_faster |= in_f == 192 && out_f == 64 && out_x == 28 && out_y == 28 && batch == 1;
    general_is_faster |= in_f == 576 && out_f == 96 && out_x == 19 && out_y == 19 && batch == 1;
    general_is_faster |= in_f == 384 && out_f == 96 && out_x == 19 && out_y == 19 && batch == 1;
    general_is_faster |= in_f == 384 && out_f == 64 && out_x == 19 && out_y == 19 && batch == 1;
    general_is_faster |= in_f == 192 && out_f == 64 && out_x == 19 && out_y == 19 && batch == 1;
    general_is_faster |= in_f == 96 && out_f == 576 && out_x == 19 && out_y == 19 && batch == 1;
    general_is_faster |= in_f == 1024 && out_f == 256 && out_x == 14 && out_y == 14 && batch == 1;
    general_is_faster |= in_f == 256 && out_f == 256 && out_x == 14 && out_y == 14 && batch == 1;
    general_is_faster |= in_f == 136 && out_f == 816 && out_x == 14 && out_y == 14 && batch == 1;
    general_is_faster |= in_f == 1280 && out_f == 256 && out_x == 10 && out_y == 10 && batch == 1;
    general_is_faster |= in_f == 256 && out_f == 128 && out_x == 3 && out_y == 3 && batch == 1;

    if (general_is_faster && !x_strided) {
        priority = FORCE_PRIORITY_3;
    }

    // Better to use kernel with 4 input features in a loop
    if (static_cast<float>(p.weights.IFM().v) / static_cast<float>(Align(p.weights.IFM().v, fsv)) < 0.5f)
        priority = FORCE_PRIORITY_4;

    return priority;
}

bool Convolution_kernel_b_fs_yx_fsv16_imad_1x1::Validate(const Params& params) const {
    if (!Parent::Validate(params)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& conv_params = *static_cast<convolution_params*>(kd.params.get());

    if ((conv_params.filterSize.x != conv_params.filterSize.y) ||
        conv_params.filterSize.x != 1) {
        // Fitler size needs to be 1x1
        return false;
    }

    if (conv_params.groups != 1)
        return false;

    if (conv_params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS) {
        if ((conv_params.activations_zero_points.empty() || conv_params.weights_zero_points.empty()) &&
            (conv_params.compensation.empty()))
            return false;
    } else if (conv_params.quantization == QuantizationType::ASYMMETRIC_DATA) {
        if ((conv_params.activations_zero_points.empty()) &&
            (conv_params.compensation.empty()))
            return false;
    } else if (conv_params.quantization == QuantizationType::ASYMMETRIC_WEIGHTS) {
        if (conv_params.weights_zero_points.empty())
            return false;
    } else {
        if (!conv_params.activations_zero_points.empty() ||
            !conv_params.weights_zero_points.empty() ||
            !conv_params.compensation.empty())
            return false;
    }

    return true;
}

WeightsLayout Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetPreferredWeightsLayout(const convolution_params& params) const {
    // TODO Auto tune index is needed in GetPreferredWeightsLayout to select correct weights layout
    auto tparams = GetAutoTuneParams(params, -1);
    if (tparams.out_block_features == 2)
        return WeightsLayout::os_is_zyx_osv32_isv16;
    if (tparams.out_block_features == 4)
        return WeightsLayout::os_is_zyx_osv64_isv16;

    return WeightsLayout::os_is_yx_osv16_isv16;
}

Convolution_kernel_b_fs_yx_fsv16_imad_1x1::AutoTuneParams
Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetAutoTuneParams(const convolution_params& params, int index) const {
    if (index >= 0 && index < static_cast<int>(all_tune_params.size())) {
        return all_tune_params[index];
    }

    size_t block_spatial = 1;
    size_t block_features = 1;
    size_t feature_slm_split = 1;
    std::string exe_mode = EXE_MODE_DEFAULT;

    size_t total_spatial = params.outputs[0].X().v * params.outputs[0].Y().v;
    // Try two features per work-item
    if (params.outputs[0].Feature().v % 32 == 0 || params.outputs[0].Feature().v > 32 * 2)
        block_features = 2;

    // Non strict inequality here leads to some regressions, ie: [1, 64, 19, 19] (*) [384, 64, 1, 1]
    bool can_split = params.weights.IFM().v > 4 * fsv;

    // Select block size in spatial dimension
    {
        size_t max_spatial = std::min(2 * simd / block_features, total_spatial);
        size_t min_efficient_spatial = 8;

        if (max_spatial <= min_efficient_spatial) {
            block_spatial = max_spatial;
        } else {
            auto minimum_params = AutoTuneParams{ min_efficient_spatial, block_features, 1, exe_mode };
            bool preserve_occupancy = EstimateOccupancy(params, minimum_params) >= 1.f;

            size_t min_overhang = max_spatial;
            size_t best_block = min_efficient_spatial;
            bool block_write_found = false;
            bool output_pad = params.outputs[0].X().pad.Total() != 0;

            for (size_t block = min_efficient_spatial; block <= max_spatial; ++block) {
                bool c_occupancy = EstimateOccupancy(params, { block, block_features, 1, exe_mode }) >= 1.f;
                auto overhang = Align(total_spatial, block) - total_spatial;
                bool c_block_write = (overhang == 0 && !output_pad) || params.outputs[0].X().v % block == 0;

                // Kernel work-around for spills/inefficient loop order
                if (can_split && !c_occupancy && block > 14 && block_features > 1)
                    break;

                if (preserve_occupancy && !c_occupancy)
                    break;

                if (overhang <= min_overhang && (!block_write_found || c_block_write)) {
                    best_block = block;
                    min_overhang = overhang;
                    block_write_found = c_block_write;
                }
            }

            block_spatial = best_block;
        }
    }

    // Try to split features using slm to increase occupancy
    {
        auto dummy_params = AutoTuneParams{ block_spatial, block_features, 1, exe_mode };
        bool enough_occupancy = EstimateOccupancy(params, dummy_params) >= 1.f;
        if (!enough_occupancy && can_split) {
            std::vector<size_t> check_split = { 4 };
            size_t ifm_blocks = CeilDiv(params.weights.IFM().v, fsv);
            for (auto split : check_split) {
                if (split > ifm_blocks)
                    break;

                auto tmp_tune = AutoTuneParams{ block_spatial, block_features, split, exe_mode };

                bool c_lws = split * simd <= params.engineInfo.maxWorkGroupSize;
                bool c_slm = EstimateSLMUsage(params, tmp_tune) <= 1.f;
                bool c_fb = block_features <= split;
                bool c_occupancy = EstimateOccupancy(params, tmp_tune) >= 1.f;

                if (c_lws && c_slm && c_fb) {
                    feature_slm_split = split;
                }

                // Increasing split will only increase memory and work-group size, don't check bigger split
                if (!c_slm || !c_lws || c_occupancy)
                    break;
            }
        }
    }

    // Occupancy is still extremely low, try to decrease spatials
    {
        auto dummy_params = AutoTuneParams{ block_spatial, block_features, feature_slm_split, exe_mode };
        constexpr float default_threshold = 5.f / 7.f;
        constexpr float split_threshold = 4.f / 7.f;
        float threshold_occupancy = feature_slm_split == 1 ? default_threshold : split_threshold;

        if (EstimateOccupancy(params, dummy_params) < threshold_occupancy && block_spatial != 1) {
            for (size_t block = block_spatial - 1; block >= 4; --block) {
                auto tmp_params = AutoTuneParams{ block, block_features, feature_slm_split, exe_mode };
                bool c_mul = total_spatial % block == 0;
                bool c_occupancy = EstimateOccupancy(params, tmp_params) >= threshold_occupancy;

                if (c_mul) {
                    block_spatial = block;
                    if (c_occupancy)
                        break;
                }
            }
        }
    }

    return AutoTuneParams{ block_spatial, block_features, feature_slm_split, exe_mode };
}

float Convolution_kernel_b_fs_yx_fsv16_imad_1x1::EstimateOccupancy(const convolution_params& params, const AutoTuneParams& tparams) const {
    size_t blocks_s = CeilDiv(params.outputs[0].X().v * params.outputs[0].Y().v, tparams.out_block_spatial);
    size_t blocks_f = CeilDiv(params.outputs[0].Feature().v, tparams.out_block_features * simd) * tparams.feature_slm_split;
    size_t block_b = params.outputs[0].Batch().v;

    auto threads = blocks_s * blocks_f * block_b;

    return static_cast<float>(threads) / static_cast<float>(params.engineInfo.maxThreadsPerDevice);
}

float Convolution_kernel_b_fs_yx_fsv16_imad_1x1::EstimateSLMUsage(const convolution_params& params, const AutoTuneParams& tparams) const {
    size_t slm_elements = tparams.out_block_spatial * tparams.out_block_features * fsv * (tparams.feature_slm_split - 1);
    size_t slm_bytes = slm_elements * BytesPerElement(GetAccumulatorType(params));

    // TODO Actual maximum slm should also depend on number of work-groups, but this is device specific
    size_t max_slm_bytes = params.engineInfo.maxLocalMemSize;

    return static_cast<float>(slm_bytes) / static_cast<float>(max_slm_bytes);
}

bool Convolution_kernel_b_fs_yx_fsv16_imad_1x1::ValidateAutoTuneParams(const convolution_params& params,
                                                                       const AutoTuneParams& tune_params) const {
    bool c_ifm = CeilDiv(params.weights.IFM().v, fsv) >= tune_params.feature_slm_split;
    bool c_slm = EstimateSLMUsage(params, tune_params) <= 1.f;
    bool c_lws = tune_params.feature_slm_split * simd <= params.engineInfo.maxWorkGroupSize;

    // Work-around for lack of actual AutoTuneParams in GetPreferredWeightsLayout
    auto default_params = GetAutoTuneParams(params, -1);
    bool c_wa_fb = default_params.out_block_features == tune_params.out_block_features;

    return c_ifm && c_slm && c_lws && c_wa_fb;
}

KernelsData Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

KernelsData Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetTunedKernelsDataByIndex(const Params & params,
                                                                                  int autoTuneIndex) const {
    auto conv_params = static_cast<const convolution_params&>(params);
    auto tune_params = GetAutoTuneParams(conv_params, autoTuneIndex);
    if (!ValidateAutoTuneParams(conv_params, tune_params))
        return {};
    return GetCommonKernelsData(params, tune_params.exe_mode, autoTuneIndex);
}

KernelsData Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetKernelsDataForAutoTune(const Params & params) const {
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

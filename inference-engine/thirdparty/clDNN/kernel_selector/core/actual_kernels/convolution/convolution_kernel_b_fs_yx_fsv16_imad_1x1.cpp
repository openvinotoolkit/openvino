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
#include <string>

//
// Kernel specific constants
//
static constexpr size_t fsv = 16;
static constexpr size_t simd = 16;

namespace kernel_selector {

Convolution_kernel_b_fs_yx_fsv16_imad_1x1::Convolution_kernel_b_fs_yx_fsv16_imad_1x1()
    : ConvolutionKernelBase("convolution_gpu_b_fs_yx_fsv16_imad_1x1") {
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
    k.DisableTuning();
    return k;
}

JitConstants Convolution_kernel_b_fs_yx_fsv16_imad_1x1::GetJitConstants(const convolution_params& params,
                                                                        const DispatchData& kd) const {
    auto mem_consts = Parent::GetJitConstants(params, kd);
    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_SPATIAL", kd.cldnnStyle.blockWidth));
    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_FEATURES", kd.cldnnStyle.blockHeight));
    mem_consts.AddConstant(MakeJitConstant("FEATURE_SLM_SPLIT", kd.cldnnStyle.prefetch));
    mem_consts.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    mem_consts.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        std::vector<std::string> idx_order = { "out_b",
                                               "(out_f + ofb * SIMD)",
                                               "intel_sub_group_shuffle(out_y_shuffle[os / SIMD], os % SIMD)",
                                               "intel_sub_group_shuffle(out_x_shuffle[os / SIMD], os % SIMD)" };
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
    DispatchData kd;
    const auto& output = params.output;
    auto tune_params = GetAutoTuneParams(params, index);
    size_t k_slices = tune_params.feature_slm_split;

    kd.gws0 = CeilDiv(output.X().v * output.Y().v, tune_params.out_block_spatial);
    kd.gws1 = CeilDiv(output.Feature().v, tune_params.out_block_features * simd) * simd * k_slices;
    kd.gws2 = output.Batch().v;

    kd.lws0 = 1;
    kd.lws1 = simd * k_slices;
    kd.lws2 = 1;

    kd.cldnnStyle = {0, 0, 0, 0, 0};
    kd.gemmStyle = {0, 0, 0, 0, 0, 0};

    kd.cldnnStyle.blockWidth = tune_params.out_block_spatial;
    kd.cldnnStyle.blockHeight = tune_params.out_block_features;
    kd.cldnnStyle.prefetch = k_slices;

    kd.efficiency = FORCE_PRIORITY_2;

    auto in_f = params.weights.IFM().v;
    auto out_f = params.weights.OFM().v;
    auto batch = output.Batch().v;
    auto out_x = output.X().v;
    auto out_y = output.Y().v;

    bool x_strided = params.stride.x != 1;
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
        kd.efficiency = FORCE_PRIORITY_3;
    }

    // Better to use kernel with 4 input features in a loop
    if (static_cast<float>(params.weights.IFM().v) / static_cast<float>(Align(params.weights.IFM().v, fsv)) < 0.5f)
        kd.efficiency = FORCE_PRIORITY_4;

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

    if (newParams.groups != 1 || newParams.split != 1)
        return false;

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
    std::string exe_mode = DEFAULT;

    size_t total_spatial = params.output.X().v * params.output.Y().v;
    // Try two features per work-item
    if (params.output.Feature().v % 32 == 0 || params.output.Feature().v > 32 * 2)
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
            bool output_pad = params.output.X().pad.Total() != 0;

            for (size_t block = min_efficient_spatial; block <= max_spatial; ++block) {
                bool c_occupancy = EstimateOccupancy(params, { block, block_features, 1, exe_mode }) >= 1.f;
                auto overhang = Align(total_spatial, block) - total_spatial;
                bool c_block_write = (overhang == 0 && !output_pad) || params.output.X().v % block == 0;

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
    size_t blocks_s = CeilDiv(params.output.X().v * params.output.Y().v, tparams.out_block_spatial);
    size_t blocks_f = CeilDiv(params.output.Feature().v, tparams.out_block_features * simd) * tparams.feature_slm_split;
    size_t block_b = params.output.Batch().v;

    auto threads = blocks_s * blocks_f * block_b;
    constexpr size_t max_threads_per_cu = 7;
    size_t compute_units = params.engineInfo.computeUnitsCount;
    size_t max_threads = compute_units * max_threads_per_cu;

    return static_cast<float>(threads) / static_cast<float>(max_threads);
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

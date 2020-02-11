// Copyright (c) 2018-2019 Intel Corporation
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


#include "convolution_kernel_b_fs_yx_fsv4_1x1.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>
#include <string>

namespace kernel_selector {

namespace {
    constexpr size_t fsv = 4;
    constexpr size_t pref_simd = 16;
    constexpr size_t pref_features_per_wi = 16;

    size_t get_preferred_lwg_depth(const DataTensor& output, const WeightsTensor& weights, const EngineInfo& info) {
        constexpr size_t threads_per_eu = 7;
        size_t max_simd_number = info.computeUnitsCount * threads_per_eu;

        size_t simd_number = CeilDiv(output.X().v * output.Y().v, pref_simd) *
                             CeilDiv(output.Feature().v, pref_features_per_wi) *
                             output.Batch().v;

        constexpr size_t preferred_ifm_multiple = 16 * 4;

        size_t ifm = weights.IFM().v;
        size_t current_lwg_depth = 1;

        while (simd_number < max_simd_number && ifm % (current_lwg_depth * 2 * preferred_ifm_multiple) == 0 &&
               current_lwg_depth < 8) {
            current_lwg_depth *= 2;
            simd_number *= 2;
        }

        return current_lwg_depth;
    }
}  // namespace

ConvolutionKernel_b_fs_yx_fsv4_1x1::ConvolutionKernel_b_fs_yx_fsv4_1x1()
    : ConvolutionKernelBase("convolution_gpu_b_fs_yx_fsv4_1x1") {
    std::vector<size_t> simd_sizes = { 16 };
    std::vector<size_t> features_per_wi = { 16, 32 };
    std::vector<size_t> lwg_depth = { 1, 2, 4, 8 };
    std::vector<std::string> exe_modes = ConvolutionKernelBase::autoTuneOptions;

    for (auto simd : simd_sizes) {
        for (auto f_wi : features_per_wi) {
            if (f_wi % simd == 0) {
                for (auto l_d : lwg_depth) {
                    for (auto exe_mode : exe_modes) {
                        all_tune_params.push_back(AutoTuneParams{ simd, f_wi, l_d, false, exe_mode });
                        all_tune_params.push_back(AutoTuneParams{ simd, f_wi, l_d, true, exe_mode });
                    }
                }
            }
        }
    }
}

ParamsKey ConvolutionKernel_b_fs_yx_fsv4_1x1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
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

bool ConvolutionKernel_b_fs_yx_fsv4_1x1::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if (newParams.filterSize.x != 1 || newParams.filterSize.y != 1)
        return false;

    // TODO: Loosen this requirement
    if (newParams.output.Feature().v % fsv != 0)
        return false;

    return true;
}

bool ConvolutionKernel_b_fs_yx_fsv4_1x1::ValidateAutoTuneParams(const convolution_params& params, const AutoTuneParams& tune_params) const {
    auto sel_lwg_d = tune_params.lwg_depth;
    if (CeilDiv(params.weights.IFM().v, fsv) % sel_lwg_d != 0) {
        return false;
    }

    return true;
}

ConvolutionKernel_b_fs_yx_fsv4_1x1::AutoTuneParams ConvolutionKernel_b_fs_yx_fsv4_1x1::GetAutoTuneParams(const convolution_params& params,
                                                                                                         int index) const {
    AutoTuneParams tune_params;
    bool selected = false;
    if (index >= 0 && index < static_cast<int>(all_tune_params.size())) {
        tune_params = all_tune_params[index];
        selected = true;
    }

    // Validate selected params
    if (selected) {
        selected = ValidateAutoTuneParams(params, tune_params);
    }

    // Set default ones
    if (!selected) {
        auto lwg_depth = get_preferred_lwg_depth(params.output, params.weights, params.engineInfo);
        tune_params = AutoTuneParams{ pref_simd, pref_features_per_wi, lwg_depth, false, DEFAULT };
    }

    return tune_params;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv4_1x1::GetJitConstants(const convolution_params& params,
                                                                 const DispatchData& kd) const {
    auto mem_consts = Parent::GetJitConstants(params, kd);

    auto simd = kd.lws0;
    auto features_per_wi = kd.cldnnStyle.blockHeight;
    auto lwg_depth = kd.lws2;
    auto force_prefetch = kd.cldnnStyle.prefetch == 1;

    mem_consts.AddConstant(MakeJitConstant("SIMD", simd));
    mem_consts.AddConstant(MakeJitConstant("FEATURES_PER_WI", features_per_wi));

    mem_consts.AddConstant(MakeJitConstant("LWG_DEPTH", lwg_depth));
    mem_consts.AddConstant(MakeJitConstant("FORCE_PREFETCH", force_prefetch));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        auto conf = FusedOpsConfiguration("",
                                          {"b", "(f + out_fi * 4)", "y", "x"},
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

ConvolutionKernelBase::DispatchData ConvolutionKernel_b_fs_yx_fsv4_1x1::SetDefault(const convolution_params& params,
                                                                                   int autoTuneIndex) const {
    DispatchData kd;
    auto& out = params.output;

    auto autoTuneParam = GetAutoTuneParams(params, autoTuneIndex);
    auto lwg_depth = autoTuneParam.lwg_depth;
    auto simd = autoTuneParam.simd;
    auto features_per_wi = autoTuneParam.features_per_wi;

    std::vector<size_t> global = { RoundUp(out.X().v * out.Y().v, simd), CeilDiv(out.Feature().v, features_per_wi), out.Batch().v * lwg_depth };
    std::vector<size_t> local = { simd, 1, lwg_depth};

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    kd.gemmStyle = { 0, 0, 0, 0, 0, 0 };

    kd.cldnnStyle.blockHeight = features_per_wi;
    kd.cldnnStyle.blockWidth = simd;
    kd.cldnnStyle.prefetch = autoTuneParam.force_prefetch ? 1 : 0;

    kd.effiency = FORCE_PRIORITY_1;

    return kd;
}  // SetDefault

KernelsData ConvolutionKernel_b_fs_yx_fsv4_1x1::GetTunedKernelsDataByIndex(const Params& params,
                                                                           const optional_params& options,
                                                                           int autoTuneIndex) const {
    auto convParams = static_cast<const convolution_params&>(params);
    auto tuneParams = GetAutoTuneParams(convParams, autoTuneIndex);
    return GetCommonKernelsData(params, options, tuneParams.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv4_1x1::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv4_1x1::GetKernelsDataForAutoTune(const Params& params,
                                                                          const optional_params& options) const {
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

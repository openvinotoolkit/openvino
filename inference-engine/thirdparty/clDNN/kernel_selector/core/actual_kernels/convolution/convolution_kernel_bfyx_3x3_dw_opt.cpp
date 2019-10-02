// Copyright (c) 2017-2018 Intel Corporation
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


#include "convolution_kernel_bfyx_3x3_dw_opt.h"
#include <vector>

namespace kernel_selector {
ConvolutionKernel_bfyx_3x3_dw_opt::ConvolutionKernel_bfyx_3x3_dw_opt()
    : ConvolutionKernelBase("convolution_gpu_bfyx_3x3_dw_opt") {
    // Generate the dispatch options to the auto-tuner.
    std::vector<size_t> tileXDimSizes = {1, 2, 4, 5, 6, 8, 10, 12, 14};
    std::vector<size_t> tileYDimSizes = {1, 2, 3, 4, 5, 6, 7};
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;

    for (auto tileXDim : tileXDimSizes) {
        for (auto tileYDim : tileYDimSizes) {
            for (auto executionMode : executionModes) {
                autoTuneOptions.emplace_back(AutoTuneOption{{tileXDim, tileYDim}, executionMode});
            }
        }
    }
}

ParamsKey ConvolutionKernel_bfyx_3x3_dw_opt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableDepthwiseSeparableOpt();
    return k;
}

bool ConvolutionKernel_bfyx_3x3_dw_opt::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }

    const convolution_params& cp = static_cast<const convolution_params&>(p);

    if ((cp.filterSize.x != 3) || (cp.filterSize.y != 3) || (cp.stride.x != 1) || (cp.stride.y != 1) ||
        (cp.padding.x != 1) || (cp.padding.y != 1) || (cp.inputs[0].Feature().v != cp.split) ||
        cp.output.PitchesDifferFromLogicalDims()) {
        return false;
    }

    return true;
}

ConvolutionKernel_bfyx_3x3_dw_opt::AutoTuneOption ConvolutionKernel_bfyx_3x3_dw_opt::GetAutoTuneOptions(
    const Params&,
    int autoTuneIndex) const {
    if ((autoTuneIndex >= 0) && (autoTuneIndex < static_cast<int>(autoTuneOptions.size()))) {
        return autoTuneOptions[autoTuneIndex];
    }

    constexpr int simdSize = 16;

    return AutoTuneOption{{simdSize - 2, 7}, DEFAULT};
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_3x3_dw_opt::SetDefault(const convolution_params& params,
                                                                                  int autoTuneIndex) const {
    constexpr int simdSize = 16;

    DispatchData runInfo = Parent::SetDefault(params);

    auto options = GetAutoTuneOptions(params, autoTuneIndex);

    const int numTilesX = static_cast<int>(
        std::ceil(static_cast<float>(params.inputs[0].X().v) / static_cast<float>(options.tileDims.x)));
    const int numTilesY = static_cast<int>(
        std::ceil(static_cast<float>(params.inputs[0].Y().v) / static_cast<float>(options.tileDims.y)));

    runInfo.cldnnStyle.blockWidth = options.tileDims.x;
    runInfo.cldnnStyle.blockHeight = options.tileDims.y;
    runInfo.gws0 = numTilesX * simdSize;
    runInfo.gws1 = numTilesY;
    runInfo.gws2 = params.inputs[0].Feature().v * params.inputs[0].Batch().v;
    runInfo.lws0 = simdSize;
    runInfo.lws1 = 1;
    runInfo.lws2 = 1;

    runInfo.effiency = FORCE_PRIORITY_5;

    return runInfo;
}

JitConstants ConvolutionKernel_bfyx_3x3_dw_opt::GetJitConstants(const convolution_params& params,
                                                                const DispatchData& kd) const {
    stSize tileDims = {kd.cldnnStyle.blockWidth, kd.cldnnStyle.blockHeight};
    auto mem_consts = ConvolutionKernelBase::GetJitConstants(params, kd);

    if (tileDims.y != 0 && tileDims.x != 0) {
        mem_consts.AddConstant(MakeJitConstant("UNIT_BYTE_SIZE", kd.fp16UnitUsed ? sizeof(short) : sizeof(float)));
        mem_consts.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", kd.lws0));
        mem_consts.AddConstant(MakeJitConstant("TILE_HEIGHT", tileDims.y));
        mem_consts.AddConstant(MakeJitConstant("TILE_WIDTH", tileDims.x));
    }

    return mem_consts;
}

KernelsData ConvolutionKernel_bfyx_3x3_dw_opt::GetTunedKernelsDataByIndex(const Params& params,
                                                                          const optional_params& options,
                                                                          const int autoTuneIndex) const {
    constexpr int simdSize = 16;

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& convParams = *static_cast<convolution_params*>(kd.params.get());
    DispatchData runInfo = SetDefault(convParams, autoTuneIndex);

    if (static_cast<int>(static_cast<int>(runInfo.gws0 - 1) / simdSize) * runInfo.cldnnStyle.blockWidth + simdSize >
        convParams.inputs[0].Y().pitch) {
        // Internal Error - requested tile size is not supported for y pitch
        return {};
    }

    return GetCommonKernelsData(params, options, GetAutoTuneOptions(params, autoTuneIndex).exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_bfyx_3x3_dw_opt::GetKernelsData(const Params& params,
                                                              const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options, -1);
}

KernelsData ConvolutionKernel_bfyx_3x3_dw_opt::GetKernelsDataForAutoTune(const Params& params,
                                                                         const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, options, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    KernelsData defaultKds = GetKernelsData(params, options);
    res.insert(res.end(), defaultKds.begin(), defaultKds.end());

    return res;
}
}  // namespace kernel_selector
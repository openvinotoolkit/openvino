// Copyright (c) 2019 Intel Corporation
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

#include "convolution_kernel_fs_byx_fsv32_1x1.h"
#include <vector>

namespace kernel_selector {

// Weights take 32 * 2 = 64 registers, max output 16 * 2 = 32 gives save 96 max register usage
static constexpr size_t maxBlockSize = 16;
static constexpr size_t subGroupSize = 16;
static constexpr size_t fsv = 32;
static constexpr size_t fsvPerThread = fsv / subGroupSize;

ConvolutionKernel_fs_byx_fsv32_1x1::ConvolutionKernel_fs_byx_fsv32_1x1()
    : ConvolutionKernelBase("convolution_gpu_fs_byx_fsv32_1x1") {
    std::vector<size_t> blockWidths = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<size_t> blockHeights = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;

    for (auto w : blockWidths) {
        for (auto h : blockHeights) {
            if (w * h <= maxBlockSize) {
                for (auto exeMode : executionModes) {
                    autoTuneOptions.emplace_back(AutoTuneOption{w, h, exeMode});
                }
            }
        }
    }
}

ParamsKey ConvolutionKernel_fs_byx_fsv32_1x1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDilation();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

ConvolutionKernel_fs_byx_fsv32_1x1::AutoTuneOption ConvolutionKernel_fs_byx_fsv32_1x1::GetAutoTuneOptions(
    const Params& arg,
    int autoTuneIndex) const {
    if (autoTuneIndex >= 0 && autoTuneIndex < static_cast<int>(autoTuneOptions.size()))
        return autoTuneOptions[autoTuneIndex];

    const convolution_params& cp = static_cast<const convolution_params&>(arg);

    ConvolutionKernel_fs_byx_fsv32_1x1::AutoTuneOption result = {1, 1, AGE_BASED};

    size_t selected_w = 0;
    size_t selected_h = 0;
    std::vector<size_t> blockSizes = {8, 7, 6, 5, 4};

    if (cp.output.X().v <= 8) {
        selected_w = cp.output.X().v;
     } else {
        for (auto w : blockSizes) {
            if (cp.output.X().v % w == 0) {
                selected_w = w;
                break;
            }
        }
    }

    if (cp.output.Y().v <= 8 && selected_w * cp.output.Y().v <= maxBlockSize) {
        selected_h = cp.output.Y().v;
    } else {
        for (auto h : blockSizes) {
            if (cp.output.Y().v % h == 0 && selected_w * h <= maxBlockSize) {
                selected_h = h;
                break;
            }
        }
    }

    if (selected_w == 0 && selected_h == 0) {
        selected_w = 8;
        selected_h = 2;
    } else if (selected_h == 0) {
        selected_h = maxBlockSize / selected_w;
    } else if (selected_w == 0) {
        selected_w = maxBlockSize / selected_h;
    }

    return {selected_w, selected_h, AGE_BASED};
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_fs_byx_fsv32_1x1::SetDefault(const convolution_params& arg,
                                                                                   int autoTuneIndex) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

    AutoTuneOption option = GetAutoTuneOptions(arg, autoTuneIndex);

    runInfo.effiency = FORCE_PRIORITY_2;

    runInfo.cldnnStyle.blockHeight = option.blockHeight;
    runInfo.cldnnStyle.blockWidth = option.blockWidth;

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = 16;

    runInfo.gws0 = CeilDiv(arg.output.X().v, option.blockWidth);
    runInfo.gws1 = CeilDiv(arg.output.Y().v, option.blockHeight);
    runInfo.gws2 = CeilDiv(arg.output.Feature().v, 32) * 16 * arg.output.Batch().v;

    return runInfo;
}

bool ConvolutionKernel_fs_byx_fsv32_1x1::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o))
        return false;

    const convolution_params& cp = static_cast<const convolution_params&>(p);

    if (cp.filterSize.x != 1 || cp.filterSize.y != 1)
        return false;

    // Output feature padding must be multiple of fsv to keep block alignment
    if (cp.output.Feature().pad.before % fsv != 0)
        return false;

    return true;
}

JitConstants ConvolutionKernel_fs_byx_fsv32_1x1::GetJitConstants(const convolution_params& params,
                                                                 const DispatchData& kd) const {
    auto jit = ConvolutionKernelBase::GetJitConstants(params, kd);

    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_WIDTH", kd.cldnnStyle.blockWidth));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_HEIGHT", kd.cldnnStyle.blockHeight));
    jit.AddConstant(MakeJitConstant("FSV", fsv));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subGroupSize));
    jit.AddConstant(MakeJitConstant("FSV_PER_THREAD", fsvPerThread));

    return jit;
}

KernelsData ConvolutionKernel_fs_byx_fsv32_1x1::GetTunedKernelsDataByIndex(const Params& params,
                                                                           const optional_params& options,
                                                                           const int autoTuneIndex) const {
    auto tuneOptions = GetAutoTuneOptions(params, autoTuneIndex);
    return GetCommonKernelsData(params, options, tuneOptions.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_fs_byx_fsv32_1x1::GetKernelsData(const Params& params,
                                                               const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

KernelsData ConvolutionKernel_fs_byx_fsv32_1x1::GetKernelsDataForAutoTune(const Params& params,
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

    return res;
}

}  // namespace kernel_selector

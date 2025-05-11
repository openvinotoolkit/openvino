// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_fs_byx_fsv32_depthwise.h"
#include <vector>

namespace kernel_selector {

static constexpr size_t subGroupSize = 16;
static constexpr size_t fsv = 32;
static constexpr size_t fsvPerThread = fsv / subGroupSize;

ConvolutionKernel_fs_byx_fsv32_depthwise::ConvolutionKernel_fs_byx_fsv32_depthwise()
    : ConvolutionKernelBase("convolution_gpu_fs_byx_fsv32_depthwise") {
    std::vector<size_t> blockWidths = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;

    for (auto w : blockWidths) {
        for (auto exeMode : executionModes) {
            autoTuneOptions.emplace_back(AutoTuneOption{w, exeMode});
        }
    }
}

ParamsKey ConvolutionKernel_fs_byx_fsv32_depthwise::GetSupportedKey() const {
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
    k.EnableGroupedConvolution();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_fs_byx_fsv32_depthwise::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

size_t ConvolutionKernel_fs_byx_fsv32_depthwise::getInputWidth(const convolution_params& arg, size_t blockWidth) const {
    return (blockWidth - 1) * arg.stride.x + (arg.filterSize.x - 1) * arg.dilation.x + 1;
}

size_t ConvolutionKernel_fs_byx_fsv32_depthwise::getMinRegisterUsage(const convolution_params& arg, size_t blockWidth) const {
    size_t weightsRegisters = 2;
    size_t outputRegisters = blockWidth * 2;
    size_t inputRegisters = getInputWidth(arg, blockWidth) * 2;

    return weightsRegisters + outputRegisters + inputRegisters;
}

ConvolutionKernel_fs_byx_fsv32_depthwise::AutoTuneOption ConvolutionKernel_fs_byx_fsv32_depthwise::GetAutoTuneOptions(
    const Params& arg,
    int autoTuneIndex) const {
    if (autoTuneIndex >= 0 && autoTuneIndex < static_cast<int>(autoTuneOptions.size()))
        return autoTuneOptions[autoTuneIndex];

    const convolution_params& cp = static_cast<const convolution_params&>(arg);

    const size_t regThreshold = 64;

    std::vector<size_t> nonOptBlockWidths = {3, 2, 1};  // This will most likely be memory bound
    std::vector<size_t> optBlockWidths = {8, 7, 6, 5, 4};

    // Check if output can be evenly divided into large blocks
    for (auto w : optBlockWidths) {
        if (cp.outputs[0].X().v % w == 0 && getMinRegisterUsage(cp, w) < regThreshold)
            return {w, EXE_MODE_AGE_BASED};
    }

    // Try to find large blocks with smallest offset
    size_t minLeftover = static_cast<size_t>(-1);
    size_t foundWidth = 0;
    for (auto w : optBlockWidths) {
        if (getMinRegisterUsage(cp, w) < regThreshold && Pad(cp.outputs[0].X().v, w) < minLeftover) {
            minLeftover = Pad(cp.outputs[0].X().v, w);
            foundWidth = w;
        }
    }

    if (foundWidth != 0)
        return {foundWidth, EXE_MODE_AGE_BASED};

    // Check small and memory bound block sizes
    for (auto w : nonOptBlockWidths) {
        if (cp.outputs[0].X().v % w == 0 && getMinRegisterUsage(cp, w) < regThreshold)
            return {w, EXE_MODE_AGE_BASED};
    }

    // This means all previous block sizes consumed too much registers, fallback to block width = 1
    return {1, EXE_MODE_AGE_BASED};
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_fs_byx_fsv32_depthwise::SetDefault(const convolution_params& arg,
                                                                                         int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(arg);

    AutoTuneOption option = GetAutoTuneOptions(arg, autoTuneIndex);

    dispatchData.cldnnStyle.blockHeight = 1;
    dispatchData.cldnnStyle.blockWidth = option.blockWidth;
    dispatchData.cldnnStyle.inputBlockWidth = getInputWidth(arg, option.blockWidth);

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 16;

    dispatchData.gws[0] = CeilDiv(arg.outputs[0].X().v, option.blockWidth);
    dispatchData.gws[1] = arg.outputs[0].Y().v;
    dispatchData.gws[2] = CeilDiv(arg.outputs[0].Feature().v, 32) * 16 * arg.outputs[0].Batch().v;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_fs_byx_fsv32_depthwise::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}

bool ConvolutionKernel_fs_byx_fsv32_depthwise::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p))
        return false;

    auto cp = static_cast<const convolution_params&>(p);
    if (cp.groups < 16)
        return false;

    if (cp.inputs[0].Feature().v != cp.groups || cp.outputs[0].Feature().v != cp.groups)
        return false;

    // Output feature padding must be multiple of fsv to keep block alignment
    if (cp.outputs[0].Feature().pad.before % fsv != 0)
        return false;

    // Input feature padding must be multiple of fsv to keep block alignment
    if (cp.inputs[0].Feature().pad.before % fsv != 0)
        return false;

    return true;
}

JitConstants ConvolutionKernel_fs_byx_fsv32_depthwise::GetJitConstants(const convolution_params& params,
                                                                       const DispatchData& dispatchData) const {
    auto jit = ConvolutionKernelBase::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("INPUT_BLOCK_WIDTH", dispatchData.cldnnStyle.inputBlockWidth));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_WIDTH", dispatchData.cldnnStyle.blockWidth));
    jit.AddConstant(MakeJitConstant("FSV", fsv));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subGroupSize));
    jit.AddConstant(MakeJitConstant("FSV_PER_THREAD", fsvPerThread));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetUnitType(params);
        FusedOpsConfiguration conf_vec_elem = {"_VEC_ELEM",
                                               {"b", "(fs * FSV + sglid + out_f * SUB_GROUP_SIZE)", "or", "oc + out_x"},
                                               "tmp_write[out_f]", input_dt, 1 };
        FusedOpsConfiguration conf_scalar = {"_SCALAR",
                                             {"b", "(fs * FSV + sglid + out_f * SUB_GROUP_SIZE)", "or", "oc + out_x"},
                                             "out[out_idx]", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec_elem, conf_scalar}));
    }

    return jit;
}

KernelsData ConvolutionKernel_fs_byx_fsv32_depthwise::GetTunedKernelsDataByIndex(const Params& params,
                                                                                 const int autoTuneIndex) const {
    auto tuneOptions = GetAutoTuneOptions(params, autoTuneIndex);
    return GetCommonKernelsData(params, tuneOptions.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_fs_byx_fsv32_depthwise::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

KernelsData ConvolutionKernel_fs_byx_fsv32_depthwise::GetKernelsDataForAutoTune(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

}  // namespace kernel_selector

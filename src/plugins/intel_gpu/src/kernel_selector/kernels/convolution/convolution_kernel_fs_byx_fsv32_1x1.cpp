// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    return k;
}

DeviceFeaturesKey ConvolutionKernel_fs_byx_fsv32_1x1::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

ConvolutionKernel_fs_byx_fsv32_1x1::AutoTuneOption ConvolutionKernel_fs_byx_fsv32_1x1::GetAutoTuneOptions(
    const Params& arg,
    int autoTuneIndex) const {
    if (autoTuneIndex >= 0 && autoTuneIndex < static_cast<int>(autoTuneOptions.size()))
        return autoTuneOptions[autoTuneIndex];

    const convolution_params& cp = static_cast<const convolution_params&>(arg);

    ConvolutionKernel_fs_byx_fsv32_1x1::AutoTuneOption result = {1, 1, EXE_MODE_AGE_BASED};

    size_t selected_w = 0;
    size_t selected_h = 0;
    std::vector<size_t> blockSizes = {8, 7, 6, 5, 4};

    if (cp.outputs[0].X().v <= 8) {
        selected_w = cp.outputs[0].X().v;
     } else {
        for (auto w : blockSizes) {
            if (cp.outputs[0].X().v % w == 0) {
                selected_w = w;
                break;
            }
        }
    }

    if (cp.outputs[0].Y().v <= 8 && selected_w * cp.outputs[0].Y().v <= maxBlockSize) {
        selected_h = cp.outputs[0].Y().v;
    } else {
        for (auto h : blockSizes) {
            if (cp.outputs[0].Y().v % h == 0 && selected_w * h <= maxBlockSize) {
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

    return {selected_w, selected_h, EXE_MODE_AGE_BASED};
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_fs_byx_fsv32_1x1::SetDefault(const convolution_params& arg,
                                                                                   int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(arg);

    AutoTuneOption option = GetAutoTuneOptions(arg, autoTuneIndex);

    dispatchData.cldnnStyle.blockHeight = option.blockHeight;
    dispatchData.cldnnStyle.blockWidth = option.blockWidth;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 16;

    dispatchData.gws[0] = CeilDiv(arg.outputs[0].X().v, option.blockWidth);
    dispatchData.gws[1] = CeilDiv(arg.outputs[0].Y().v, option.blockHeight);
    dispatchData.gws[2] = CeilDiv(arg.outputs[0].Feature().v, 32) * 16 * arg.outputs[0].Batch().v;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_fs_byx_fsv32_1x1::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}

bool ConvolutionKernel_fs_byx_fsv32_1x1::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p))
        return false;

    const convolution_params& cp = static_cast<const convolution_params&>(p);

    if (cp.filterSize.x != 1 || cp.filterSize.y != 1)
        return false;

    // Output feature padding must be multiple of fsv to keep block alignment
    if (cp.outputs[0].Feature().pad.before % fsv != 0)
        return false;

    // Input feature padding must be multiple of fsv to keep block alignment
    if (cp.inputs[0].Feature().pad.before % fsv != 0)
        return false;

    return true;
}

JitConstants ConvolutionKernel_fs_byx_fsv32_1x1::GetJitConstants(const convolution_params& params,
                                                                 const DispatchData& dispatchData) const {
    auto jit = ConvolutionKernelBase::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_WIDTH", dispatchData.cldnnStyle.blockWidth));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_HEIGHT", dispatchData.cldnnStyle.blockHeight));
    jit.AddConstant(MakeJitConstant("FSV", fsv));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subGroupSize));
    jit.AddConstant(MakeJitConstant("FSV_PER_THREAD", fsvPerThread));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetUnitType(params);
        FusedOpsConfiguration conf_vec_elem = {"_VEC_ELEM",
                                               {"b", "(fs * FSV + sglid + out_f * SUB_GROUP_SIZE)", "or + out_y", "oc + out_x"},
                                               "tmp_write[out_f]", input_dt, 1 };
        FusedOpsConfiguration conf_scalar = {"_SCALAR",
                                             {"b", "(fs * FSV + sglid + out_f * SUB_GROUP_SIZE)", "or + out_y", "oc + out_x"},
                                             "out[out_idx]", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec_elem, conf_scalar}));
    }

    return jit;
}

KernelsData ConvolutionKernel_fs_byx_fsv32_1x1::GetTunedKernelsDataByIndex(const Params& params,
                                                                           const int autoTuneIndex) const {
    auto tuneOptions = GetAutoTuneOptions(params, autoTuneIndex);
    return GetCommonKernelsData(params, tuneOptions.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_fs_byx_fsv32_1x1::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

KernelsData ConvolutionKernel_fs_byx_fsv32_1x1::GetKernelsDataForAutoTune(const Params& params) const {
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

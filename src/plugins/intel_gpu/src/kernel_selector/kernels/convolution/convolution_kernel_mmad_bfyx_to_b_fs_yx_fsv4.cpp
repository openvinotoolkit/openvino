// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_mmad_bfyx_to_b_fs_yx_fsv4.h"
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <iostream>

namespace kernel_selector {

ParamsKey ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv4::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv4::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_blocked_read_write();
    k.requires_blocked_read_write_short();

    return k;
}

bool ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv4::Validate(const Params &p) const {
    if (!Parent::Validate(p)) {
        return false;
    }

    auto params = dynamic_cast<const convolution_params&>(p);

    if (params.inputs[0].Feature().v != 3)
        return false;

    return true;
}

ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv4::AutoTuneOption ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv4::GetAutoTuneOptions(const Params &p,
                                                                                                                        int autoTuneIndex) const {
    if ((autoTuneIndex >= 0) && (autoTuneIndex < static_cast<int>(autoTuneOptions.size()))) {
        return autoTuneOptions[autoTuneIndex];
    }

    AutoTuneOption option = {0, 0, 0, EXE_MODE_DEFAULT};

    auto &params = dynamic_cast<const convolution_params &>(p);
    auto &output = params.outputs[0];

    // TODO: Check if other block size can improve performance
    option.blockHeight = 1;
    option.prefetch = 1;
    if (output.LogicalSize() < 49 * 1024) {
        option.blockWidth = 4;
    } else {
        option.blockWidth = 8;
    }

    return option;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv4::SetDefault(const convolution_params &cp,
                                                                                            int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(cp);

    auto tuneOptions = GetAutoTuneOptions(cp, autoTuneIndex);
    dispatchData.cldnnStyle.blockWidth = tuneOptions.blockWidth;
    dispatchData.cldnnStyle.blockHeight = tuneOptions.blockHeight;
    dispatchData.cldnnStyle.prefetch = tuneOptions.prefetch;

    dispatchData.gws[0] = Align(cp.outputs[0].Feature().v, 32) / 2;
    dispatchData.gws[1] = CeilDiv(cp.outputs[0].X().v, dispatchData.cldnnStyle.blockWidth) * cp.outputs[0].Y().v;
    dispatchData.gws[2] = cp.outputs[0].Batch().v;

    dispatchData.lws[0] = 16;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv4::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

JitConstants ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv4::GetJitConstants(const convolution_params &params,
                                                                        const DispatchData &dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", dispatchData.lws[0]));
    jit.AddConstant(MakeJitConstant("OSV", 32));
    jit.AddConstant(MakeJitConstant("ISV", 32));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", dispatchData.cldnnStyle.blockWidth));
    jit.AddConstant(MakeJitConstant("IFM_BLOCKS", CeilDiv(params.inputs[0].Feature().v, 32)));
    auto input = params.inputs[0];
    auto output = params.outputs[0];
    auto blockWidth = dispatchData.cldnnStyle.blockWidth;
    size_t input_line_size = std::min(params.stride.x * (blockWidth - 1) + (params.weights.X().v - 1) * params.dilation.x + 1,
                                      input.X().v + input.X().pad.Total());

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));

    jit.Merge(MakeTypeJitConstants(GetPackedInputType(params), "PACKED_IN"));
    jit.Merge(MakeTypeJitConstants(GetPackedType(params.outputs[0].GetDType(), 2), "PACKED_OUT"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf0 = {"_0", {"b", "(fg*32 + 2*lid+0)", "y", "(x+i)"}, "res0", input_dt, 1};
        FusedOpsConfiguration conf1 = {"_1", {"b", "(fg*32 + 2*lid+1)", "y", "(x+i)"}, "res1", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf0, conf1}));
    }

    return jit;
}

KernelsData ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv4::GetKernelsData(const Params &params) const {
    KernelsData kd = GetTunedKernelsDataByIndex(params);
    return kd;
}

KernelsData ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv4::GetKernelsDataForAutoTune(const Params &params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelsData res = {};

    for (int i = -1; i < static_cast<int>(autoTuneOptions.size()); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

}  // namespace kernel_selector

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_mmad_b_fs_yx_fsv32.h"
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <iostream>

namespace kernel_selector {

ParamsKey ConvolutionKernel_mmad_b_fs_yx_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_WEIGHTS);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_mmad_b_fs_yx_fsv32::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_blocked_read_write();

    return k;
}

bool ConvolutionKernel_mmad_b_fs_yx_fsv32::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        return false;
    }

    auto params = dynamic_cast<const convolution_params&>(p);

    if ((params.quantization == QuantizationType::ASYMMETRIC_DATA || params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS)
        && !params.HasCompensation()) {
        return false;
    }

    if (!IsSIMDSizeSupported(params.engineInfo, 8))
        return false;

    if (params.groups > 1)
        return false;

    return true;
}

ConvolutionKernel_mmad_b_fs_yx_fsv32::AutoTuneOption ConvolutionKernel_mmad_b_fs_yx_fsv32::GetAutoTuneOptions(const Params& p,
                                                                                                              int autoTuneIndex) const {
    if ((autoTuneIndex >= 0) && (autoTuneIndex < static_cast<int>(autoTuneOptions.size()))) {
        return autoTuneOptions[autoTuneIndex];
    }

    AutoTuneOption option = {0, 0, 0, EXE_MODE_DEFAULT};

    auto& params = dynamic_cast<const convolution_params&>(p);
    auto& output = params.outputs[0];

    // TODO: Check if other block size can improve performance
    option.blockHeight = 1;
    option.prefetch = 1;
    if (output.LogicalSize() < 49 * 1024)
        option.blockWidth = 4;
    else
        option.blockWidth = 8;

    return option;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_mmad_b_fs_yx_fsv32::SetDefault(const convolution_params& cp,
                                                                                     int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(cp);

    auto tuneOptions = GetAutoTuneOptions(cp, autoTuneIndex);
    dispatchData.cldnnStyle.blockWidth = tuneOptions.blockWidth;
    dispatchData.cldnnStyle.blockHeight = tuneOptions.blockHeight;
    dispatchData.cldnnStyle.prefetch = tuneOptions.prefetch;

    size_t ow_group = 8;
    while (ow_group > 1) {
        if (CeilDiv(cp.outputs[0].X().v, dispatchData.cldnnStyle.blockWidth) % ow_group == 0)
            break;
        ow_group--;
    }

    dispatchData.gws[0] = Align(cp.outputs[0].Feature().v, 32) / 4;
    dispatchData.gws[1] = Align(CeilDiv(cp.outputs[0].X().v, dispatchData.cldnnStyle.blockWidth), ow_group) * cp.outputs[0].Y().v * cp.outputs[0].Z().v;
    dispatchData.gws[2] = cp.outputs[0].Batch().v;

    dispatchData.lws[0] = 8;
    dispatchData.lws[1] = ow_group;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_mmad_b_fs_yx_fsv32::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

JitConstants ConvolutionKernel_mmad_b_fs_yx_fsv32::GetJitConstants(const convolution_params& params,
                                                                   const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("OW_GROUP", dispatchData.lws[1]));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", dispatchData.lws[0]));
    jit.AddConstant(MakeJitConstant("OSV_SIZE", 32));
    jit.AddConstant(MakeJitConstant("ISV_SIZE", 32));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", dispatchData.cldnnStyle.blockWidth));
    jit.AddConstant(MakeJitConstant("IFM_BLOCKS", CeilDiv(params.inputs[0].Feature().v, 32)));
    auto input = params.inputs[0];
    auto output = params.outputs[0];
    auto blockWidth = dispatchData.cldnnStyle.blockWidth;
    size_t input_line_size = params.stride.x * (blockWidth - 1) + (params.weights.X().v - 1)*params.dilation.x + 1;

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));

    jit.Merge(MakeTypeJitConstants(GetPackedInputType(params), "PACKED_IN"));
    jit.Merge(MakeTypeJitConstants(GetPackedOutputType(params), "PACKED_OUT"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);

        std::vector<std::string> idx_order0;
        std::vector<std::string> idx_order1;
        std::vector<std::string> idx_order2;
        std::vector<std::string> idx_order3;
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
            idx_order0 = {"b", "(fg*32 + 4*lid+0)", "y", "(x+i)"};
            idx_order1 = {"b", "(fg*32 + 4*lid+1)", "y", "(x+i)"};
            idx_order2 = {"b", "(fg*32 + 4*lid+2)", "y", "(x+i)"};
            idx_order3 = {"b", "(fg*32 + 4*lid+3)", "y", "(x+i)"};
        } else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
            idx_order0 = {"b", "(fg*32 + 4*lid+0)", "z", "y", "(x+i)"};
            idx_order1 = {"b", "(fg*32 + 4*lid+1)", "z", "y", "(x+i)"};
            idx_order2 = {"b", "(fg*32 + 4*lid+2)", "z", "y", "(x+i)"};
            idx_order3 = {"b", "(fg*32 + 4*lid+3)", "z", "y", "(x+i)"};
        }

        FusedOpsConfiguration conf0 = {"_0", idx_order0, "res0", input_dt, 1 };
        FusedOpsConfiguration conf1 = {"_1", idx_order1, "res1", input_dt, 1 };
        FusedOpsConfiguration conf2 = {"_2", idx_order2, "res2", input_dt, 1 };
        FusedOpsConfiguration conf3 = {"_3", idx_order3, "res3", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf0, conf1, conf2, conf3}));
    }

    return jit;
}

KernelsData ConvolutionKernel_mmad_b_fs_yx_fsv32::GetKernelsData(const Params& params) const {
    KernelsData kd = GetTunedKernelsDataByIndex(params);
    return kd;
}

KernelsData ConvolutionKernel_mmad_b_fs_yx_fsv32::GetKernelsDataForAutoTune(const Params& params) const {
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

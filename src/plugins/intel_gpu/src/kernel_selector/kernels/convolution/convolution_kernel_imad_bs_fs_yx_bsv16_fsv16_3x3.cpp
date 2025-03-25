// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_3x3.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>
#include <iostream>

//
// Kernel specific constants
//
#define SIMD_SIZE 16

namespace kernel_selector {

ParamsKey Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_3x3::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);

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

DeviceFeaturesKey Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_3x3::get_required_device_features_key(const Params&) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_shuffle();

    return k;
}

KernelsData Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_3x3::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

JitConstants Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_3x3::GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const {
    auto mem_consts = Parent::GetJitConstants(params, dispatchData);

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = {"",
                                             {"out_b", "out_f + get_sub_group_local_id()", "out_y", "out_x"},
                                             "dequantized",
                                             input_dt,
                                             1,
                                             LoadType::FEATURE_SHUFFLE};
        conf_scalar.SetLoopAxes({ Tensor::DataChannelName::BATCH }, true);
        conf_scalar.SetShuffleVarName("i");
        mem_consts.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return mem_consts;
}  // GetJitConstants

ConvolutionKernelBase::DispatchData Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_3x3::SetDefault(const convolution_params& params, int) const {
    DispatchData dispatchData;
    const auto& output = params.outputs[0];

    dispatchData.gws = { output.X().v, output.Y().v, output.Feature().v / 16 * output.Batch().v };
    dispatchData.lws = { 1, 1, SIMD_SIZE };

    dispatchData.cldnnStyle = {0, 0, 0, 0, 0};
    dispatchData.gemmStyle = {0, 0, 0, 0, 0, 0};

    return dispatchData;
}  // SetDefault

KernelsPriority Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_3x3::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

bool Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_3x3::Validate(const Params& params) const {
    if (!Parent::Validate(params)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if ((newParams.filterSize.x != newParams.filterSize.y) ||
        newParams.filterSize.x != 3) {
        // Fitler size needs to be 3x3
        return false;
    }

    if (newParams.stride.x != newParams.stride.y) {
        // Strides must be equal
        return false;
    }
    if (newParams.outputs[0].X().v != newParams.outputs[0].Y().v) {
        // W and H must be equal
        return false;
    }

    if (newParams.outputs[0].Feature().v % 16 != 0) {
        // output feature size must be divided by 16
        return false;
    }

    if (newParams.outputs[0].Batch().v % 16 != 0) {
        // batch size must be divided by 16
        return false;
    }

    // check that all fused ops except eltwise have only feature or scalar inputs
    for (auto& fo : newParams.fused_ops) {
        if (fo.GetType() == FusedOpType::ELTWISE)
            continue;
        for (auto& input : fo.tensors) {
            if (input.X().v != 1 || input.Y().v != 1 || input.Batch().v != 1)
                return false;
        }
    }

    return true;
}
}  // namespace kernel_selector

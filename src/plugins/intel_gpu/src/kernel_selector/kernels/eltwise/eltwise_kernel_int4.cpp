// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation/activation_kernel_base.h"
#include "eltwise_kernel_int4.h"
#include "kernel_selector_utils.h"
#include <string>
#include <algorithm>

namespace kernel_selector {

ParamsKey EltwiseKernel_int4::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT4);
    k.EnableInputDataType(Datatype::UINT4);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
//    k.EnableAllInputLayout();
//    k.EnableAllOutputLayout();
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    return k;
}

JitConstants EltwiseKernel_int4::GetJitConstants(const eltwise_params& params) const {
    return GetJitConstantsCommon(params, true);
}

bool EltwiseKernel_int4::Validate(const Params& params) const {
    if (!EltwiseKernelBase::Validate(params)) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);

    // Only one activation can be fused.
    if (ewParams.fused_ops.size() > 1 ||
        (ewParams.activations.size() !=0 && ewParams.fused_ops.size() != 0)) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    /*
        for (size_t i = 0; i < ewParams.inputs.size(); i++) {
            const auto input_layout = ewParams.inputs[i].GetLayout();
            const auto batch_size = ewParams.inputs[i].Batch().v;
            const auto feature_size = ewParams.inputs[i].Feature().v;
            if ((input_layout == DataLayout::b_fs_yx_fsv16 && feature_size % 16 != 0) ||
                (input_layout == DataLayout::b_fs_yx_fsv32 && feature_size % 32 != 0) ||
                (input_layout == DataLayout::b_fs_zyx_fsv16 && feature_size % 16 != 0) ||
                (input_layout == DataLayout::b_fs_yx_fsv4 && feature_size % 8 != 0) ||
                input_layout == DataLayout::fs_b_yx_fsv32 ||
                (input_layout == DataLayout::bs_fs_yx_bsv32_fsv16 && (feature_size % 16 != 0 || batch_size % 32 != 0)) ||
                (input_layout == DataLayout::bs_fs_yx_bsv32_fsv32 && (feature_size % 32 != 0 || batch_size % 32 != 0)))
                DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
        if ((ewParams.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 && ewParams.outputs[0].Feature().v % 16 != 0) ||
            (ewParams.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv32 && ewParams.outputs[0].Feature().v % 32 != 0) ||
            (ewParams.outputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv16 && ewParams.outputs[0].Feature().v % 16 != 0) ||
            (ewParams.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv4 && ewParams.outputs[0].Feature().v % 8 != 0) ||
            ewParams.outputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32 ||
            (ewParams.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 &&
                (ewParams.outputs[0].Feature().v % 16 != 0 || ewParams.outputs[0].Batch().v % 32 != 0)) ||
            (ewParams.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32 &&
                (ewParams.outputs[0].Feature().v % 32 != 0 || ewParams.outputs[0].Batch().v % 32 != 0)))
            DO_NOT_USE_THIS_KERNEL(params.layerID);

    const auto& output = ewParams.outputs[0];
    const auto count = output.PhysicalSize();

    const bool bSupportedCount = (count % 8) == 0;

    bool bCheckSizes = true;
    for (size_t i = 0; i < ewParams.inputs.size(); i++) {
        // allow only the same input sizes or scalars, without pitches
        if (ewParams.inputs[i].PitchesDifferFromLogicalDims() ||
            (!(ewParams.inputs[0] == ewParams.inputs[i] && ewParams.inputs[i] == ewParams.outputs[0]) &&
             ewParams.inputs[i].PhysicalSize() != 1))
            bCheckSizes = false;
    }

    // TODO: add support to this implementation when user requests input values updates
    bool bCheckUpdateInput = true;
    if (!ewParams.updateInputIds.empty())
        bCheckUpdateInput = false;

    // TODO: add support for reading from output buffer and using its values in computation
    bool bCheckUseOutput = true;
    for (size_t op = 0; op < ewParams.operations.size(); op++) {
        for (size_t input_idx = 0; input_idx < ewParams.operations[op].inputs.size(); input_idx++) {
            if (ewParams.operations[op].inputs[input_idx].mode == EltwiseInputMode::OUTPUT_BUFFER) {
                bCheckUseOutput = false;
                break;
            }
        }
    }

    if (IsUnsupportedModeForVecCode(ewParams))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (!bCheckSizes || !bSupportedCount || !bCheckUpdateInput || !bCheckUseOutput) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }
            */

    //printf("!!!! in eltwise validate done \n");
    return true;
}

KernelsData EltwiseKernel_int4::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority EltwiseKernel_int4::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}
}  // namespace kernel_selector

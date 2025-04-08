// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "concatenation_kernel_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

size_t getTileXY(const concatenation_params& params) {
    auto& input = params.inputs[0];
    size_t tileXY =  1;
    if (params.isAligned) {
        switch (input.GetDType()) {
        case Datatype::F16:
        case Datatype::INT8:
        case Datatype::UINT8:
            tileXY = 4;
            break;
        default:
            return 1;
        }
    } else {
        switch (input.GetDType()) {
        case Datatype::F32:
            tileXY = 2;
            break;
        case Datatype::F16:
            tileXY = 4;
            break;
        case Datatype::INT8:
        case Datatype::UINT8:
            tileXY = 8;
            break;
        default:
            return 1;
        }
    }

    auto tileXYMultiple = input.X().v;
    bool noInputPad = input.X().pad.Total() == 0;
    bool noOutputPad = params.outputs[0].X().pad.Total() == 0;
    if (noInputPad && noOutputPad)
        tileXYMultiple = input.X().v * input.Y().v;

    while (tileXYMultiple % tileXY != 0)
        tileXY /= 2;

    return tileXY;
}

}  // namespace

ParamsKey ConcatenationKernel_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableConcatAxis(ConcatAxis::FEATURE);
    k.EnableConcatKernelPerInput();
    return k;
}

DeviceFeaturesKey ConcatenationKernel_b_fs_yx_fsv16::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle_relative();

    return k;
}

bool ConcatenationKernel_b_fs_yx_fsv16::Validate(const Params& p) const {
    if (!ConcatenationKernelBase::Validate(p)) {
        return false;
    }

    const concatenation_params& params = static_cast<const concatenation_params&>(p);

    // all inputs have to have same layout
    auto same_layout = params.inputs[0].GetLayout();
    for (const auto& lt : params.inputs) {
        if (lt.GetLayout() != same_layout) {
            return false;
        }
    }

    if (params.axis != ConcatAxis::FEATURE)
        return false;

    return true;
}

ConcatenationKernelBase::DispatchData ConcatenationKernel_b_fs_yx_fsv16::SetDefault(const concatenation_params& params) const {
    DispatchData dispatchData = ConcatenationKernelBase::SetDefault(params);
    const auto& input = params.inputs[0];
    auto tileXY = getTileXY(params);

    size_t tileF = params.misalignment == 0 ? 1 : 2;

    dispatchData.gws[0] = CeilDiv(input.X().v * input.Y().v, tileXY);
    dispatchData.gws[1] = Align(input.Feature().v, 16 * tileF) / tileF;
    dispatchData.gws[2] = input.Batch().v;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 16;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConcatenationKernel_b_fs_yx_fsv16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

JitConstants ConcatenationKernel_b_fs_yx_fsv16::GetJitConstants(const concatenation_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("ALIGNED", params.misalignment == 0));
    jit.AddConstant(MakeJitConstant("MISALIGNMENT", params.misalignment));
    jit.AddConstant(MakeJitConstant("TILE_XY", getTileXY(params)));

    return jit;
}

KernelsData ConcatenationKernel_b_fs_yx_fsv16::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

size_t ConcatenationKernel_b_fs_yx_fsv16::GetAlignment(const concatenation_params& /*params*/) const {
    return 16;
}
}  // namespace kernel_selector

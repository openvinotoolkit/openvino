// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenation_kernel_simple_ref.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

ParamsKey ConcatenationKernel_simple_Ref::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::fyxb);
    k.EnableOutputLayout(DataLayout::fyxb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableConcatAxis(ConcatAxis::X);
    k.EnableConcatAxis(ConcatAxis::Y);
    k.EnableConcatAxis(ConcatAxis::Z);
    k.EnableConcatAxis(ConcatAxis::W);
    k.EnableConcatAxis(ConcatAxis::FEATURE);
    k.EnableConcatAxis(ConcatAxis::BATCH);
    k.EnableConcatKernelPerInput();
    k.EnableDifferentTypes();
    return k;
}

bool ConcatenationKernel_simple_Ref::Validate(const Params& p, const optional_params& o) const {
    if (!ConcatenationKernelBase::Validate(p, o)) {
        return false;
    }

    const concatenation_params& params = static_cast<const concatenation_params&>(p);

    // all inputs have to have same layout (exept 3D: bfzyx, b_fs_zyx_fsv16, and bs_fs_zyx_bsv16_fsv16)
    auto same_layout = params.inputs[0].GetLayout();
    for (const auto& lt : params.inputs) {
        auto cur_layout = lt.GetLayout();
        if ((cur_layout == DataLayout::bfzyx || cur_layout == DataLayout::b_fs_zyx_fsv16 || cur_layout == DataLayout::bs_fs_zyx_bsv16_fsv16) &&
            (same_layout == DataLayout::bfzyx || same_layout == DataLayout::b_fs_zyx_fsv16 || same_layout == DataLayout::bs_fs_zyx_bsv16_fsv16
            || same_layout == DataLayout::bs_fs_yx_bsv32_fsv32)) {
            continue;
        } else if (cur_layout != same_layout) {
            return false;
        }
    }

    return true;
}

ConcatenationKernelBase::DispatchData ConcatenationKernel_simple_Ref::SetDefault(const concatenation_params& params) const {
    DispatchData dispatchData;
    const auto& input = params.inputs[0];

    dispatchData.gws = { input.X().v * input.Y().v,
                         input.Z().v * input.W().v,
                         input.Feature().v * input.Batch().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData ConcatenationKernel_simple_Ref::GetKernelsData(const Params& params, const optional_params& optParams) const {
    KernelsData kd = GetCommonKernelsData(params, optParams);
    return kd;
}

KernelsPriority ConcatenationKernel_simple_Ref::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector

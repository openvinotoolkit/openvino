// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_blocked_single_axis.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey SoftmaxKernelBlockedSingleAxis::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);

    k.EnableSoftmaxDim(SoftmaxDim::X);
    k.EnableSoftmaxDim(SoftmaxDim::Y);
    k.EnableSoftmaxDim(SoftmaxDim::Z);
    k.EnableSoftmaxDim(SoftmaxDim::FEATURE);
    k.EnableSoftmaxDim(SoftmaxDim::BATCH);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();

    return k;
}

SoftmaxKernelBlockedSingleAxis::DispatchData SoftmaxKernelBlockedSingleAxis::SetDefault(const softmax_params& params,
                                                                                                const optional_params& optParams) const {
    auto dispatchData = Parent::SetDefault(params, optParams);
    const auto& out = params.outputs[0];

    switch (params.dim) {
        case SoftmaxDim::X:
            dispatchData.gws = {out.Y().v * out.Z().v, out.Feature().v, out.Batch().v};
            break;
        case SoftmaxDim::Y:
            dispatchData.gws = {out.X().v * out.Z().v, out.Feature().v, out.Batch().v};
            break;
        case SoftmaxDim::Z:
            dispatchData.gws = {out.X().v * out.Y().v, out.Feature().v, out.Batch().v};
            break;
        case SoftmaxDim::FEATURE:
            dispatchData.gws = {out.X().v * out.Z().v, out.Y().v, out.Batch().v};
            break;
        case SoftmaxDim::BATCH:
            dispatchData.gws = {out.X().v * out.Z().v, out.Y().v, out.Feature().v};
            break;
        default:
            dispatchData.gws = {1, 1, 1};
    }

    dispatchData.lws = {1, 1, 1};

    return dispatchData;
}

KernelsPriority SoftmaxKernelBlockedSingleAxis::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool SoftmaxKernelBlockedSingleAxis::Validate(const Params& params, const optional_params& o) const {
    if (!Parent::Validate(params, o)) {
        return false;
    }
    const auto& softmax_params = static_cast<const kernel_selector::softmax_params&>(params);
    return softmax_params.dim == SoftmaxDim::BATCH
            || softmax_params.dim == SoftmaxDim::FEATURE
            || softmax_params.dim == SoftmaxDim::Z
            || softmax_params.dim == SoftmaxDim::Y
            || softmax_params.dim == SoftmaxDim::X;
}

KernelsData SoftmaxKernelBlockedSingleAxis::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
JitConstants SoftmaxKernelBlockedSingleAxis::GetJitConstants(const softmax_params& params, DispatchData dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    jit.AddConstant(MakeJitConstant("SOFTMAX_DIM_" + toString(params.dim), "1"));

    std::vector<std::string> idx_order;
    const auto ndims = params.inputs[0].GetDims().size();
    switch (params.dim) {
        case SoftmaxDim::X:
            jit.AddConstant(MakeJitConstant("CLASS_NUM", "INPUT0_SIZE_X"));
            idx_order = {"other3", "other1", ndims == 5 ? "other2" : "0", "other0", "cls"};
            break;
        case SoftmaxDim::Y:
            jit.AddConstant(MakeJitConstant("CLASS_NUM", "INPUT0_SIZE_Y"));
            idx_order = {"other3", "other1", ndims == 5 ? "other2" : "0", "cls", "other0"};
            break;
        case SoftmaxDim::Z:
            jit.AddConstant(MakeJitConstant("CLASS_NUM", "INPUT0_SIZE_Z"));
            idx_order = {"other3", "other1", "cls", "other2", "other0"};
            break;
        case SoftmaxDim::FEATURE:
            jit.AddConstant(MakeJitConstant("CLASS_NUM", "INPUT0_FEATURE_NUM"));
            idx_order = {"other3", "cls", ndims == 5 ? "other2" : "0", "other1", "other0"};
            break;
        case SoftmaxDim::BATCH:
            jit.AddConstant(MakeJitConstant("CLASS_NUM", "INPUT0_BATCH_NUM"));
            idx_order = {"cls", "other3", ndims == 5 ? "other2" : "0", "other1", "other0"};
            break;
        default:
            break;
    }

    auto acc_dt = GetActivationType(params);
    jit.Merge(MakeTypeJitConstants(acc_dt, "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"",
                                      idx_order,
                                      "res",
                                      acc_dt};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}
}  // namespace kernel_selector
